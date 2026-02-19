import torch
import torch.nn as nn
import torch.nn.functional as F


class SSLLoss(nn.Module):
    """
    SSL (Self-Supervised Learning) Loss using NVIDIA NeMo SSL models.
    Computes cosine similarity loss between embeddings of clean and enhanced audio.
    """
    
    def __init__(self, model_name="nvidia/ssl_en_nest_xlarge_v1.0", device="cuda"):
        super(SSLLoss, self).__init__()
        self.device = device
        self.model_name = model_name
        
        # Load NVIDIA SSL model from NeMo
        try:
            import nemo.collections.asr as nemo_asr
            print(f"Loading SSL model: {model_name}")
            self.ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.from_pretrained(
                model_name=model_name
            )
            self.ssl_model = self.ssl_model.to(device)
            self.ssl_model.eval()
            
            # Freeze SSL model parameters
            for param in self.ssl_model.parameters():
                param.requires_grad = False
                
            print(f"SSL model loaded successfully: {sum(p.numel() for p in self.ssl_model.parameters()):,} params")
            
        except Exception as e:
            print(f"Error loading SSL model: {e}")
            raise e
    
    def get_embeddings(self, audio):
        """
        Extract embeddings from audio using the SSL model.
        
        Args:
            audio: Tensor of shape [B, T] - audio waveform at 16kHz
            
        Returns:
            embeddings: Tensor of shape [B, T', D] - frame-level embeddings
        """
        with torch.no_grad():
            # Get audio lengths
            audio_lens = torch.tensor([audio.shape[1]] * audio.shape[0], device=self.device)
            
            # Get embeddings from the SSL model encoder
            # The SSL model expects audio and audio_lens
            processed_signal, processed_signal_length = self.ssl_model.preprocessor(
                input_signal=audio,
                length=audio_lens
            )
            
            # Get encoder output
            encoded, encoded_len = self.ssl_model.encoder(
                audio_signal=processed_signal,
                length=processed_signal_length
            )
            
        return encoded
    
    def forward(self, clean_audio, enhanced_audio):
        """
        Compute SSL loss between clean and enhanced audio.
        
        Args:
            clean_audio: Tensor [B, T] - clean reference audio
            enhanced_audio: Tensor [B, T] - enhanced/denoised audio
            
        Returns:
            loss: Scalar tensor - cosine similarity loss
        """
        # Ensure same length
        min_len = min(clean_audio.shape[1], enhanced_audio.shape[1])
        clean_audio = clean_audio[:, :min_len]
        enhanced_audio = enhanced_audio[:, :min_len]
        
        # Get embeddings
        clean_emb = self.get_embeddings(clean_audio)
        enhanced_emb = self.get_embeddings(enhanced_audio.detach() if not enhanced_audio.requires_grad else enhanced_audio)
        
        # Re-enable gradients for enhanced embeddings computation
        # We need to recompute with gradients
        processed_signal, processed_signal_length = self.ssl_model.preprocessor(
            input_signal=enhanced_audio,
            length=torch.tensor([enhanced_audio.shape[1]] * enhanced_audio.shape[0], device=self.device)
        )
        enhanced_emb, _ = self.ssl_model.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length
        )
        
        # Align temporal dimensions
        min_time = min(clean_emb.shape[1], enhanced_emb.shape[1])
        clean_emb = clean_emb[:, :min_time, :]
        enhanced_emb = enhanced_emb[:, :min_time, :]
        
        # Compute cosine similarity loss (1 - cosine_similarity)
        # Normalize embeddings
        clean_emb_norm = F.normalize(clean_emb, p=2, dim=-1)
        enhanced_emb_norm = F.normalize(enhanced_emb, p=2, dim=-1)
        
        # Cosine similarity per frame
        cos_sim = (clean_emb_norm * enhanced_emb_norm).sum(dim=-1)  # [B, T']
        
        # Loss = 1 - mean cosine similarity
        loss = 1.0 - cos_sim.mean()
        
        return loss


class SSLLossSimple(nn.Module):
    """
    Simplified SSL Loss that uses mean pooled embeddings.
    More stable for training.
    """
    
    def __init__(self, model_name="nvidia/ssl_en_nest_xlarge_v1.0", device="cuda"):
        super(SSLLossSimple, self).__init__()
        self.device = device
        self.model_name = model_name
        
        try:
            import nemo.collections.asr as nemo_asr
            print(f"Loading SSL model: {model_name}")
            self.ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.from_pretrained(
                model_name=model_name
            )
            self.ssl_model = self.ssl_model.to(device)
            self.ssl_model.eval()
            
            for param in self.ssl_model.parameters():
                param.requires_grad = False
                
            print(f"SSL model loaded: {sum(p.numel() for p in self.ssl_model.parameters()):,} params")
            
        except Exception as e:
            print(f"Error loading SSL model: {e}")
            raise e
    
    def forward(self, clean_audio, enhanced_audio):
        """
        Compute SSL loss using mean-pooled embeddings.
        """
        min_len = min(clean_audio.shape[1], enhanced_audio.shape[1])
        clean_audio = clean_audio[:, :min_len]
        enhanced_audio = enhanced_audio[:, :min_len]
        
        audio_lens = torch.tensor([min_len] * clean_audio.shape[0], device=self.device)
        
        # Get clean embeddings (no grad needed)
        with torch.no_grad():
            clean_processed, clean_len = self.ssl_model.preprocessor(
                input_signal=clean_audio,
                length=audio_lens
            )
            clean_emb, _ = self.ssl_model.encoder(
                audio_signal=clean_processed,
                length=clean_len
            )
        
        # Get enhanced embeddings (with grad for backprop through generator)
        enhanced_processed, enhanced_len = self.ssl_model.preprocessor(
            input_signal=enhanced_audio,
            length=audio_lens
        )
        enhanced_emb, _ = self.ssl_model.encoder(
            audio_signal=enhanced_processed,
            length=enhanced_len
        )
        
        # Align temporal dimensions
        min_time = min(clean_emb.shape[1], enhanced_emb.shape[1])
        clean_emb = clean_emb[:, :min_time, :]
        enhanced_emb = enhanced_emb[:, :min_time, :]
        
        # Mean pool over time
        clean_emb_pooled = clean_emb.mean(dim=1)  # [B, D]
        enhanced_emb_pooled = enhanced_emb.mean(dim=1)  # [B, D]
        
        # DEBUG: Print embedding stats (remove after debugging)
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 5:
            print(f"[DEBUG SSL] clean_emb shape: {clean_emb.shape}, mean: {clean_emb.mean().item():.6f}, std: {clean_emb.std().item():.6f}")
            print(f"[DEBUG SSL] enhanced_emb shape: {enhanced_emb.shape}, mean: {enhanced_emb.mean().item():.6f}, std: {enhanced_emb.std().item():.6f}")
            print(f"[DEBUG SSL] diff norm: {(clean_emb_pooled - enhanced_emb_pooled).norm().item():.6f}")
            self._debug_count += 1
        # L2 loss on embeddings
        cos_sim = F.cosine_similarity(enhanced_emb_pooled, clean_emb_pooled, dim=-1)
        loss = (1.0 - cos_sim).mean()

        return loss
