# Defining the dataset to be used
DATA="mpeg7"

# Number of pairs
N_PAIRS=1000

# Training set split
SPLIT=0.5

# Batch size
BATCH_SIZE=32

# Learning rate
LR=0.001

# Epochs
EPOCHS=100

# Number of runnings
N_RUNS=1

# Creating a loop
for i in $(seq 1 $N_RUNS); do
    # Learns the similarities
    python learn_similarities.py $DATA $N_PAIRS -tr_split $SPLIT -batch_size $BATCH_SIZE -lr $LR -epochs $EPOCHS -seed $i

    # Moves Dualing's convergence file
    mv dualing.log outputs/${DATA}_${i}_convergence.txt

    # Classifies with OPF
    python classify_with_opf.py $DATA -tr_split $SPLIT -seed $i --use_similarity

    # Processes the report
    python process_report.py $DATA -seed $i --use_similarity
done