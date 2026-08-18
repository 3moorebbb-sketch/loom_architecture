import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy.stats as stats
import copy

# =====================================================================
# V6.0 PREREGISTRATION PROTOCOL: KINEMATIC DRAG & VISCOSITY
# Translating fluid 3D mesh physics into Latent Continual Learning
# =====================================================================

SEEDS = 10
EPOCHS_PER_TASK = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.01

MEMORY_BUDGET_SIZE = 50  

# Kinematic Physics Parameters
DRAG_COEFFICIENT = 0.05  # How fast the tail follows the head (0 = frozen, 1 = instant)
VISCOSITY_TENSION = 10.0 # How hard the tail pulls back on the head

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.x_data = []
        self.y_targets = []

    def add_batch(self, x, y):
        for i in range(x.size(0)):
            if len(self.x_data) < self.capacity:
                self.x_data.append(x[i].clone().detach())
                self.y_targets.append(y[i].clone().detach())

    def sample(self, batch_size):
        if len(self.x_data) == 0: return None, None
        indices = np.random.choice(len(self.x_data), min(batch_size, len(self.x_data)), replace=False)
        return torch.stack([self.x_data[i] for i in indices]), torch.stack([self.y_targets[i] for i in indices])

def calculate_harmonic_mean(acc_a, acc_b):
    if acc_a == 0 or acc_b == 0: return 0.0
    return 2 * (acc_a * acc_b) / (acc_a + acc_b)

def print_rigorous_statistics(name, mep_scores, baseline_scores):
    mep_mean = np.mean(mep_scores)
    base_mean = np.mean(baseline_scores)
    _, p_val = stats.ttest_ind(mep_scores, baseline_scores, equal_var=False)
    
    print(f"\n--- Statistical Adjudication: V6.0 MEP vs {name} ---")
    print(f"MEP Kinematic Drag Mean: {mep_mean:.2f}%")
    print(f"Baseline ER Mean:        {base_mean:.2f}%")
    print(f"Difference:              {(mep_mean - base_mean):+.2f}%")
    print(f"Welch's t-test p-value:  {p_val:.4e} {'(Significant)' if p_val < 0.05 else '(Not Significant)'}")

class DeepLatentCNN(nn.Module):
    def __init__(self):
        super(DeepLatentCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def extract_latent_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.extract_latent_features(x))

def update_kinematic_tail(head_model, tail_model, drag):
    """
    Translating the Ethereal Loom math:
    Tail = Tail + (Head - Tail) * drag
    """
    with torch.no_grad():
        for head_param, tail_param in zip(head_model.parameters(), tail_model.parameters()):
            # EMA glide
            tail_param.data.mul_(1 - drag).add_(head_param.data * drag)

def get_split_cifar10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    def f(ds, classes):
        idx = [i for i, t in enumerate(ds.targets) if t in classes][:1500]
        return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=BATCH_SIZE, shuffle=True)
    return f(trainset, [0,1,2,3,4]), f(testset, [0,1,2,3,4]), f(trainset, [5,6,7,8,9]), f(testset, [5,6,7,8,9])

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            _, pred = torch.max(model(data).data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    return 100.0 * correct / total

def run_crucible():
    print("==================================================")
    print("V6.0 CRUCIBLE: TOPOLOGICAL VISCOSITY (KINEMATIC DRAG)")
    print("==================================================\n")

    train_A, test_A, train_B, test_B = get_split_cifar10()
    results_mep, results_er = [], []

    for seed in range(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n[Seed {seed+1}/{SEEDS}] Training Base Task A...")
        
        base_model = DeepLatentCNN()
        criterion = nn.CrossEntropyLoss()
        
        # --- TRAIN TASK A ---
        model_A = copy.deepcopy(base_model)
        opt_A = optim.SGD(model_A.parameters(), lr=LEARNING_RATE)
        for _ in range(EPOCHS_PER_TASK):
            for data, target in train_A:
                opt_A.zero_grad()
                criterion(model_A(data), target).backward()
                opt_A.step()
                
        # --- EXTRACT BUDGET (50 points) ---
        buffer_A = ReplayBuffer(MEMORY_BUDGET_SIZE)
        x_anchors = []
        for data, target in train_A:
            buffer_A.add_batch(data, target)
            for i in range(data.size(0)):
                if len(x_anchors) < MEMORY_BUDGET_SIZE: x_anchors.append(data[i].clone())
            if len(x_anchors) >= MEMORY_BUDGET_SIZE: break
        x_anchors = torch.stack(x_anchors)

        # --- TEST 1: Experience Replay Baseline ---
        model_er = copy.deepcopy(model_A)
        opt_er = optim.SGD(model_er.parameters(), lr=LEARNING_RATE)
        for _ in range(EPOCHS_PER_TASK):
            for data, target in train_B:
                opt_er.zero_grad()
                loss = criterion(model_er(data), target)
                x_buf, y_buf = buffer_A.sample(16)
                if x_buf is not None: loss += criterion(model_er(x_buf), y_buf)
                loss.backward()
                opt_er.step()
                
        er_h = calculate_harmonic_mean(evaluate(model_er, test_A), evaluate(model_er, test_B))
        results_er.append(er_h)

        # --- TEST 2: Kinematic Drag (Viscous Manifold) ---
        head_model = copy.deepcopy(model_A)
        tail_model = copy.deepcopy(model_A) # The Tail starts exactly where Task A ended
        opt_mep = optim.SGD(head_model.parameters(), lr=LEARNING_RATE)
        
        for _ in range(EPOCHS_PER_TASK):
            for data, target in train_B:
                opt_mep.zero_grad()
                
                # 1. Standard Learning on the Head
                loss_B = criterion(head_model(data), target)
                
                # 2. Topological Viscosity: The Head feels tension from the Tail
                head_features = head_model.extract_latent_features(x_anchors)
                with torch.no_grad():
                    tail_features = tail_model.extract_latent_features(x_anchors)
                
                viscosity_penalty = F.mse_loss(head_features, tail_features)
                
                loss = loss_B + (VISCOSITY_TENSION * viscosity_penalty)
                loss.backward()
                opt_mep.step()

                # 3. The Tail follows the Head (Kinematic Drag)
                update_kinematic_tail(head_model, tail_model, DRAG_COEFFICIENT)
                
        mep_h = calculate_harmonic_mean(evaluate(head_model, test_A), evaluate(head_model, test_B))
        results_mep.append(mep_h)
        
        print(f"  -> ER Harmonic:  {er_h:.2f}%")
        print(f"  -> MEP Harmonic: {mep_h:.2f}%")

    print_rigorous_statistics("Experience Replay Baseline", results_mep, results_er)

if __name__ == "__main__":
    run_crucible()