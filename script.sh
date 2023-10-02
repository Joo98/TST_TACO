time=`date +%y-%m-%d-%T`

local_rank=2
train=true 
test=false 

model='T5'
tokenizer='t5-base'
batch_size=16
model_path='/HDD/juhyoung'


for string in ${TASK} ${model} ${time}; do
    model_path+=${string}
    model_path+='_'
done
model_path+='.pt'
echo ${model_path}



if ${train} -eq true; then
    CUDA_VISIBLE_DEVICES=${local_rank} python3 run.py --training --model ${model} --tokenizer ${tokenizer} --model_path ${model_path} --batch_size ${batch_size}
    > /HDD/juhyoung/taco/log/train/_${model}_${time}.log
fi

if ${test} -eq true; then
    CUDA_VISIBLE_DEVICES=${local_rank} python3 run.py --testing --model ${model} --tokenizer ${tokenizer} --model_path ${model_path} --batch_size ${batch_size}
    > /HDD/juhyoung/taco/log/test/_${model}_${time}.log
fi