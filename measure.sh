TIMESTAMP=$(date +%Y%m%d_%H%M%S)
docker build -t energy .

#sudo chgrp -R msr /dev/cpu/*/msr
#sudo chmod g+r /dev/cpu/*/msr
#cargo build -r
#sudo setcap cap_sys_rawio=ep target/release/energibridge


/home/enrique/EnergiBridge/target/release/energibridge -g -o results/measurement_${TIMESTAMP}.csv -- \
    docker run --rm --gpus all -v "$(pwd)/results:/app/results" energy \
    --model "deepseek-ai/deepseek-coder-1.3b-base" \
    --batch ${TIMESTAMP} \
    --sleep 30




#docker run --rm --gpus all -v ./results:/app/results energy --model "deepseek-ai/deepseek-coder-1.3b-base" --batch 1 --sleep 60
