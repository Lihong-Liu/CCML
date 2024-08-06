import torch.nn as nn
import torch


class CCML(nn.Module):
    def __init__(self, num_views, dims, num_classes, device):
        super(CCML, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList([EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])
        self.device = device

    def Evidence_DC(self, alpha, beta):
        E = dict()
        for v in range(len(alpha)):
            E[v] = alpha[v]-1
            E[v] = torch.nan_to_num(E[v], 0)

        for v in range(len(alpha)):
            E[v] = torch.nan_to_num(E[v], 0)

        E_con = E[0]
        for v in range(1, len(alpha)):
            E_con = torch.min(E_con, E[v])
        for v in range(len(alpha)):
            E[v] = torch.sub(E[v], E_con)
        alpha_con = E_con + 1

        E_div = E[0]
        for v in range(1,len(alpha)):
            E_div = torch.add(E_div, E[v])

        E_div = torch.div(E_div, len(alpha))

        S_con = torch.sum(alpha_con, dim=1, keepdim=True)

        b_con = torch.div(E_con, S_con)
        S_b = torch.sum(b_con, dim=1, keepdim=True)

        b_con2 = torch.pow(b_con, beta)
        S_b2 = torch.sum(b_con2,dim=1, keepdim=True)

        b_cona = torch.mul(b_con2, torch.div(S_b, S_b2))

        E_con = torch.mul(b_cona, S_con)

        E_con = torch.mul(E_con, len(alpha))
        E_a = torch.add(E_con, E_div)

        alpha_a = E_a + 1
        alpha_con = E_con + 1

        alpha_a = torch.nan_to_num(alpha_a, 0)
        alpha_con = torch.nan_to_num(alpha_con, 0)
        alpha_div = torch.nan_to_num(E_div+1, 0)

        Sum = torch.sum(alpha_a, dim=1, keepdim=True)
        return alpha_a, alpha_con, alpha_div

    def forward(self, X, beta):
        evidences = dict()
        for v in range(self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidences[v_num] + 1
        alpha_a, alpha_con, alpha_div = self.Evidence_DC(alpha, beta)
        evidence_a = alpha_a - 1
        evidence_con = alpha_con - 1
        evidence_div = alpha_div - 1

        return evidences, evidence_a, evidence_con, evidence_div


class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h
