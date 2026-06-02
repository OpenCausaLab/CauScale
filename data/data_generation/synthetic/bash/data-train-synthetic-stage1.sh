#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/../generate_data_continuous.py"

mechanisms=("linear" "nn" "nn_add")
nodes=(10 20 100)
degrees=(1.0 2.0 3.0 4.0)
nb_observations=1
output_folder="."
output_dir="${PROJECT_ROOT}/data/datasets/synthetic/train"
dag_types=("sf" "er")

# Concurrency control
MAX_JOBS=10  # Max parallel jobs; adjust based on your CPU core count

# Counters
total=0
current=0
running_jobs=0

# Count total configurations
for mechanism in "${mechanisms[@]}"; do
    for node in "${nodes[@]}"; do
        for degree in "${degrees[@]}"; do
            for dag_type in "${dag_types[@]}"; do
                ((total++))
            done
        done
    done
done

echo "Total configurations: $total, max parallel jobs: $MAX_JOBS"

# Wait for a job slot to open
wait_for_jobs() {
    while [ $running_jobs -ge $MAX_JOBS ]; do
        wait -n  # Wait for any background job to finish
        ((running_jobs--))
    done
}

# Run a single task
run_task() {
    local mechanism=$1
    local node=$2
    local degree=$3
    local dag_type=$4
    local task_id=$5

    echo "[$task_id/$total] Start: $mechanism, $node nodes, degree $degree, dag_type $dag_type"

    python "${PYTHON_SCRIPT}" \
        --mechanism $mechanism \
        --nb-nodes $node \
        --expected-degree $degree \
        --nb-points 1000 \
        --nb-dag 600 \
        --intervention \
        --nb-interventions $node \
        --max-nb-target 1 \
        --min-nb-target 1 \
        --cover \
        --obs-data \
        --unable_verbose \
        --start-idx 0 \
        --dag-type $dag_type \
        --output_folder $output_folder \
        --output-dir $output_dir \
        --nb-observations $nb_observations \
        --task_id $task_id \
        --total $total
    # echo "[$task_id/$total] Done: $mechanism, $node nodes, degree $degree, dag_type $dag_type"
}

# Run all combinations in parallel
for mechanism in "${mechanisms[@]}"; do
    for node in "${nodes[@]}"; do
        for degree in "${degrees[@]}"; do
            for dag_type in "${dag_types[@]}"; do
                ((current++))

                # Wait for an available job slot
                wait_for_jobs

                # Launch background task
                run_task "$mechanism" "$node" "$degree" "$dag_type" "$current" &
                ((running_jobs++))
            done
        done
    done
done

# Wait for all tasks to finish
echo "Waiting for all tasks to complete..."
wait

echo "All configurations generated."
