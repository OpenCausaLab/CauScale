SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

DIR_CONFIG_BASE="${SCRIPT_DIR}/../config/train"
DIR_OUTPUT="${PROJECT_ROOT}/data/datasets/sergio/train"
BASE_CMD="python -B -m avici.experiment.data"

N_PARALLEL=20

for n_vars in 10 20 30 50 80 100 150 200; do
    DIR_CONFIG="${DIR_CONFIG_BASE}/n${n_vars}"

    for g in "sf" "er" "sbm" "sf_indirect" "sf_in"; do
        for e in 2 4 6; do
            file="data_${g}_e${e}"

            for j in {0..199}; do
                $BASE_CMD --j $j --data_config_path ${DIR_CONFIG}/${file}.yaml --path_data ${DIR_OUTPUT}/n${n_vars}/${file:5}_add --descr ${file} &

                # throttle parallel jobs
                [[ $(jobs -r -p | wc -l) -ge $N_PARALLEL ]] && wait -n
            done
        done
    done
done

wait
echo "all done"
