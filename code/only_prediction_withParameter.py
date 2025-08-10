import torch
import numpy as np
import os
import argparse
from sklearn import metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ConvNet_test(torch.nn.Module):
    def __init__(self, nummotif, motiflen, poolType, neuType, mode, learning_steps, learning_rate, learning_Momentum,
                 sigmaConv, dropprob, sigmaNeu, beta1, beta2, beta3, reverse_complemet_mode):
        super(ConvNet_test, self).__init__()
        self.poolType = poolType
        self.neuType = neuType
        self.mode = mode
        self.learning_rate = learning_rate
        self.reverse_complemet_mode = reverse_complemet_mode
        self.momentum_rate = learning_Momentum
        self.sigmaConv = sigmaConv
        self.wConv = torch.randn(nummotif, 4, motiflen).to(device)
        torch.nn.init.normal_(self.wConv, mean=0, std=self.sigmaConv)
        self.wConv.requires_grad = True
        self.wRect = torch.randn(nummotif).to(device)
        torch.nn.init.normal_(self.wRect)
        self.wRect = -self.wRect
        self.wRect.requires_grad = True
        self.dropprob = dropprob
        self.sigmaNeu = sigmaNeu
        self.wHidden = torch.randn(2 * nummotif, 32).to(device)
        self.wHiddenBias = torch.randn(32).to(device)
        self.wNeu = torch.randn(32, 1).to(device)
        self.wNeuBias = torch.randn(1).to(device)
        self.wHidden.requires_grad = True
        self.wHiddenBias.requires_grad = True
        self.wNeu.requires_grad = True
        self.wNeuBias.requires_grad = True
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

    def forward_pass(self, x, mask=None, use_mask=False):
        conv = torch.nn.functional.conv1d(x, self.wConv, bias=self.wRect, stride=1, padding=0)
        rect = conv.clamp(min=0)
        maxPool, _ = torch.max(rect, dim=2)
        if self.poolType == 'maxavg':
            avgPool = torch.mean(rect, dim=2)
            pool = torch.cat((maxPool, avgPool), 1)
        else:
            pool = maxPool
        if self.neuType == 'nohidden':
            if self.mode == 'training':
                if not use_mask:
                    mask = bernoulli.rvs(self.dropprob, size=len(pool[0]))
                    mask = torch.from_numpy(mask).float().to(device)
                pooldrop = pool * mask
                out = pooldrop @ self.wNeu
                out.add_(self.wNeuBias)
            else:
                out = self.dropprob * (pool @ self.wNeu)
                out.add_(self.wNeuBias)
        else:
            hid = pool @ self.wHidden
            hid.add_(self.wHiddenBias)
            hid = hid.clamp(min=0)
            if self.mode == 'training':
                if not use_mask:
                    mask = bernoulli.rvs(self.dropprob, size=len(hid[0]))
                    mask = torch.from_numpy(mask).float().to(device)
                hiddrop = hid * mask
                out = self.dropprob * (hid @ self.wNeu)
                out.add_(self.wNeuBias)
            else:
                out = self.dropprob * (hid @ self.wNeu)
                out.add_(self.wNeuBias)
        return out, mask

    def forward(self, x):
        if not self.reverse_complemet_mode:
            out, _ = self.forward_pass(x)
        else:
            x1, x2 = self.divide_two_tensors(x)
            out1, mask = self.forward_pass(x1)
            out2, _ = self.forward_pass(x2, mask, True)
            out = torch.max(out1, out2)
        return out


def parse_args():
    parser = argparse.ArgumentParser(description="预测转录因子模型")
    parser.add_argument('--tf_name', type=str, default='HNF4G', help="转录因子名称")
    parser.add_argument('--model_path', type=str, default='data/myExperiment/HNF4G/model/MyModel_2.pth', help="模型路径")
    parser.add_argument('--fasta_file', type=str, required=True, help="FASTA 文件路径")
    parser.add_argument('--output_file', type=str, required=True, help="输出文件路径")
    parser.add_argument('--motiflen', type=int, default=24, help="Motif长度")
    parser.add_argument('--threshold', type=float, default=0.5, help="预测的阈值")
    return parser.parse_args()


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model = ConvNet_test(16, 24, 'maxavg', 'hidden', 'test', 20000, 0.0119, 0.977, 1.26e-05, 0.75, 2.22e-04, 6.61e-10, 1.83e-09, 1.88e-06, False).to(device)
    model.wConv = checkpoint['conv']
    model.wRect = checkpoint['rect']
    model.wHidden = checkpoint['wHidden']
    model.wHiddenBias = checkpoint['wHiddenBias']
    model.wNeu = checkpoint['wNeu']
    model.wNeuBias = checkpoint['wNeuBias']
    return model


def seqtopad(sequence, motlen, kind='DNA'):
    rows = len(sequence) + 2 * motlen - 2
    S = np.empty([rows, 4])
    base = 'ACGT' if kind == 'DNA' else 'ACGU'
    for i in range(rows):
        for j in range(4):
            if i - motlen + 1 < len(sequence) and sequence[i - motlen + 1] == 'N' or i < motlen - 1 or i > len(sequence) + motlen - 2:
                S[i, j] = np.float32(0.25)
            elif sequence[i - motlen + 1] == base[j]:
                S[i, j] = np.float32(1)
            else:
                S[i, j] = np.float32(0)
    return np.transpose(S)


def predict_sequences(model, sequences, motiflen=24, device=device, threshold=0.5):
    predictions = []
    for seq_id, seq in sequences:
        seq_padded = seqtopad(seq, motiflen)
        seq_tensor = torch.from_numpy(seq_padded).unsqueeze(0).to(device)
        seq_tensor = seq_tensor.float()
        with torch.no_grad():
            model.mode = 'test'
            output = model(seq_tensor)
            pred_sig = torch.sigmoid(output)
            pred_value = pred_sig.item()
            is_specific = pred_value >= threshold
        predictions.append((seq_id, pred_value))
    return predictions


def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        seq_id = None
        seq_content = []
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if seq_id is not None:
                    sequences.append((seq_id, ''.join(seq_content)))
                seq_id = line[1:]
                seq_content = []
            else:
                seq_content.append(line)
        if seq_id is not None:
            sequences.append((seq_id, ''.join(seq_content)))
    return sequences


def save_predictions(predictions, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for seq_id, pred_value in predictions:
            f.write(f"{seq_id}\t{pred_value:.4f}\n")
    print(f"predict result save to: {output_file}")


def main():
    args = parse_args()

    model = load_model(args.model_path)

    sequences = read_fasta(args.fasta_file)

    predictions = predict_sequences(model, sequences, motiflen=args.motiflen, threshold=args.threshold)

    save_predictions(predictions, args.output_file)

if __name__ == "__main__":
    main()
