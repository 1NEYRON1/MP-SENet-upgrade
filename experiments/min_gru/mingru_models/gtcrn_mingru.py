"""
GTCRN-MinGRU: GTCRN model with MinGRU instead of GRU.
"""

from .min_gru import MinGRU
from models.gtcrn.model import GTCRN
from models.gtcrn.components import (
    TRA,
    GTConvBlock,
    GRNN,
    DPGRNN,
    Encoder,
    Decoder,
)


class MinTRA(TRA):
    def __init__(self, channels):
        super().__init__(channels)
        self.att_gru = MinGRU(channels, 2 * channels, 1, batch_first=True)


class MinGRNN(GRNN):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
    ):
        super().__init__(
            input_size, hidden_size, num_layers, batch_first, bidirectional
        )
        self.rnn1 = MinGRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.rnn2 = MinGRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )


class MinDPGRNN(DPGRNN):
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super().__init__(input_size, width, hidden_size, **kwargs)
        self.intra_rnn = MinGRNN(
            input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True
        )

        self.inter_rnn = MinGRNN(
            input_size=input_size, hidden_size=hidden_size, bidirectional=False
        )


class MinGTConvBlock(GTConvBlock):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        use_deconv=False,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            use_deconv,
        )
        self.tra = MinTRA(in_channels // 2)


class MinEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.en_convs[2] = MinGTConvBlock(
            16,
            16,
            (3, 3),
            stride=(1, 1),
            padding=(0, 1),
            dilation=(1, 1),
            use_deconv=False,
        )
        self.en_convs[3] = MinGTConvBlock(
            16,
            16,
            (3, 3),
            stride=(1, 1),
            padding=(0, 1),
            dilation=(2, 1),
            use_deconv=False,
        )
        self.en_convs[4] = MinGTConvBlock(
            16,
            16,
            (3, 3),
            stride=(1, 1),
            padding=(0, 1),
            dilation=(5, 1),
            use_deconv=False,
        )


class MinDecoder(Decoder):
    def __init__(self):
        super().__init__()
        self.de_convs[0] = MinGTConvBlock(
            16,
            16,
            (3, 3),
            stride=(1, 1),
            padding=(2 * 5, 1),
            dilation=(5, 1),
            use_deconv=True,
        )
        self.de_convs[1] = MinGTConvBlock(
            16,
            16,
            (3, 3),
            stride=(1, 1),
            padding=(2 * 2, 1),
            dilation=(2, 1),
            use_deconv=True,
        )
        self.de_convs[2] = MinGTConvBlock(
            16,
            16,
            (3, 3),
            stride=(1, 1),
            padding=(2 * 1, 1),
            dilation=(1, 1),
            use_deconv=True,
        )


class MinGTCRN(GTCRN):
    """
    GTCRN with MinGRU: All GRU components replaced with MinGRU.
    """

    def __init__(self):
        super().__init__()
        self.encoder = MinEncoder()

        self.dpgrnn1 = MinDPGRNN(16, 33, 16)
        self.dpgrnn2 = MinDPGRNN(16, 33, 16)

        self.decoder = MinDecoder()
