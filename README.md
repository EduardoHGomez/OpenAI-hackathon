📦 Your Complete Package:
├── START_HERE.md         ← You are here!
├── CHECKLIST.md          ← Follow this step-by-step
├── README_AWS.md         ← Detailed AWS instructions
├── aws_setup.sh          ← Setup script (run on AWS)
├── test_gpu.py           ← Test GPU works
├── triton_matmul.py      ← The Triton kernel (core!)
└── optimize_simple.py    ← Simple optimizer

ssh peb7frixsqxzpo-6441150a@ssh.runpod.io -i ~/.ssh/id_ed25519

Quick Commands Reference
# Generate key
ssh-keygen -t ed25519 -C "server-github" -f ~/.ssh/github_ci -N ""

# Configure SSH
cat >> ~/.ssh/config <<'EOF'
Host github.com
  User git
  IdentityFile ~/.ssh/github_ci
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

# Test
ssh -T git@github.com

# Convert existing repo
cd /path/to/<REPO>
git remote set-url origin git@github.com:<OWNER>/<REPO>.git
git fetch --all
git pull origin <BRANCH>

# Fresh clone
git clone git@github.com:<OWNER>/<REPO>.git
cd <REPO>
git pull origin <BRANCH>

Example (your repo)
# Convert remote for this repo
cd /path/to/hackathon-prep
git remote set-url origin git@github.com:EduardoHGomez/hackathon-prep.git
git pull origin main





🧠 What "autotuning" means

When you run code on a GPU, there are many ways to configure it — like how big each block of threads is, how memory is divided, or how many “warps” (groups of threads) are used.
These settings massively affect speed, but the best configuration depends on your hardware and the size of your data.

Autotuning = automatically testing different configurations to find the fastest one.

So instead of you manually guessing what block size or tile size gives the best performance, the program does trial and error — then records which setup runs fastest.