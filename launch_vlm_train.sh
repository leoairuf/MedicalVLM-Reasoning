#!/usr/bin/env bash
#SBATCH -A SNIC2022-5-277                
#SBATCH -p alvis                         
#SBATCH -N 1 --gpus-per-node=A100fat:1   
#SBATCH -t 7-00:00:00                    
#SBATCH --error=medical_vlm_reasoner.err 
#SBATCH --output=medical_vlm_reasoner.out 

# --- Carica i moduli necessari ---
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# --- Entra nella directory di lavoro ---
cd /mimer/NOBACKUP/groups/snic2022-5-277/lfuria_2/lfuria/MedicalVLM-Reasoning || exit

# --- Attiva il virtualenv (path relativo dalla cartella corrente) ---
source /mimer/NOBACKUP/groups/snic2022-5-277/lfuria_2/lfuria/vlm_train/bin/activate

# --- Lancia il tuo script Python ---
python visual_grpo.py

# --- Disattiva il virtualenv ---
deactivate
