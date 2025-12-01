import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import nemo.collections.asr as nemo_asr
    HAS_NEMO = True
except ImportError:
    HAS_NEMO = False

class SSLLoss(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base-plus", device="cuda"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.is_nemo = "nvidia" in model_name.lower() or "nest" in model_name.lower()
        self.success = False

        if self.is_nemo:
            if not HAS_NEMO:
                print("Warning: 'nemo_toolkit[asr]' is not installed, but an NVIDIA model was requested. Install it to use NEST.")
                return
            try:
                # Загрузка модели NeMo
                if "nest" in model_name.lower():
                    self.model = nemo_asr.models.EncDecDenoiseMaskedTokenPredModel.from_pretrained(model_name=model_name)
                else:
                    self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)
                
                self.model.to(device)
                self.model.eval()
                self.model.freeze() 
                self.success = True
                print(f"Successfully loaded NeMo model: {model_name}")
            except Exception as e:
                print(f"Error loading NeMo model {model_name}: {e}")
                self.success = False

        else:
            if not HAS_TRANSFORMERS:
                print("Warning: 'transformers' is not installed.")
                return
            try:
                # Используем FeatureExtractor вместо AutoProcessor
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(device)
                self.model.eval()
                
                for param in self.model.parameters():
                    param.requires_grad = False
                self.success = True
                print(f"Successfully loaded HF model: {model_name}")
            except Exception as e:
                print(f"Error loading HF model {model_name}: {e}")
                self.success = False

    def forward(self, clean_wav, gen_wav, sr=16000):
        if not self.success:
            return torch.tensor(0.0, device=self.device)

        # Приводим вход к нужному виду
        if clean_wav.dim() == 3:
             clean_wav = clean_wav.squeeze(1)
        if gen_wav.dim() == 3:
             gen_wav = gen_wav.squeeze(1)

        # --- Логика для NeMo ---
        if self.is_nemo:
            input_lengths = torch.tensor([clean_wav.shape[1]] * clean_wav.shape[0], device=self.device)
            
            with torch.no_grad():
                # 1. Preprocessing (Audio -> Mel Spectrogram)
                processed_clean, processed_clean_len = self.model.preprocessor(
                    input_signal=clean_wav, length=input_lengths
                )
                
                # 2. Encoding (Mel Spectrogram -> Embeddings)
                # Важно: encoder возвращает tuple (encoded, encoded_len)
                clean_encoded, _ = self.model.encoder(audio_signal=processed_clean, length=processed_clean_len)
                
            
            # Для сгенерированного аудио
            # Используем препроцессор (он обычно дифференцируемый или замороженный)
            # Если препроцессор (STFT) содержит операции, не поддерживающие градиенты при freeze, может потребоваться no_grad и тут,
            # но тогда мы потеряем градиенты для генератора.
            # NeMo препроцессоры (AudioToMelSpectrogramPreprocessor) - это nn.Module с torch.stft, он дифференцируем.
            
            processed_gen, processed_gen_len = self.model.preprocessor(
                input_signal=gen_wav, length=input_lengths
            )
            
            gen_encoded, _ = self.model.encoder(audio_signal=processed_gen, length=processed_gen_len)
            
            # NeMo: [B, D, T] -> [B, T, D]
            clean_emb = clean_encoded.transpose(1, 2)
            gen_emb = gen_encoded.transpose(1, 2)

        # --- Логика для Transformers ---
        else:
            with torch.no_grad():
                # Важно: WavLM/Wav2Vec2 ожидают [B, T].
                # Если feature_extractor имеет нормализацию, ее можно применить, но обычно достаточно подать raw audio.
                # Для надежности можно использовать feature_extractor, если он делает что-то важное (например, zero-mean unit-variance).
                # Но часто feature_extractor возвращает numpy, что ломает граф.
                # Просто подаем тензоры.
                clean_outputs = self.model(clean_wav, output_hidden_states=True)
                clean_emb = clean_outputs.last_hidden_state
            
            gen_outputs = self.model(gen_wav, output_hidden_states=True)
            gen_emb = gen_outputs.last_hidden_state

        # --- Общий расчет Loss ---
        # Приводим к [Batch*Frames, Dim] для CosineEmbeddingLoss
        
        if clean_emb.shape[1] != gen_emb.shape[1]:
            min_len = min(clean_emb.shape[1], gen_emb.shape[1])
            clean_emb = clean_emb[:, :min_len, :]
            gen_emb = gen_emb[:, :min_len, :]

        b, t, d = clean_emb.shape
        clean_emb_flat = clean_emb.reshape(-1, d)
        gen_emb_flat = gen_emb.reshape(-1, d)
        
        target = torch.ones(clean_emb_flat.size(0)).to(self.device)
        
        loss_fn = nn.CosineEmbeddingLoss()
        loss = loss_fn(clean_emb_flat, gen_emb_flat, target)
        
        return loss
