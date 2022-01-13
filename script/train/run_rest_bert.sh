#!/bin/bash

# hidden=210
idx=1
start=1
end=10000

bert_lr=(5e-5 3e-5) #  3e-5
for(( i=0;i<${#bert_lr[@]};i++))  #${#array[@]}获取数组长度用于循环
do
  lr=.000100
  while [ `expr $lr \> .000001` -eq 1 ]
  do
    l2reg=.000010
    while [ `expr $l2reg \> .000001` -eq 1 ]
    do
      num_layers=2
      while [ $num_layers -le 2 ]
      do
        head_num=4
        while [ $head_num -le 4 ]
        do
          gcn_dropout=.10
          while [ `expr .90 \> $gcn_dropout` -eq 1 ]
          do
            mhsa_dropout=.10
            while [ `expr .90 \> $mhsa_dropout` -eq 1 ]
            do
              threshold=.50
              while [ `expr .60 \> $threshold` -eq 1 ]
              do
                syn_srd=1
                while [ $syn_srd -le 1 ]
                do
                  sem_srd=3
                  while [ $sem_srd -le 3 ]
                  do
                    if [ $idx -ge $start ] && [ $idx -le $end ];then
                        printf "idx:%d lr:%1.5f bert_lr:%1.6f l2reg:%1.5f num_layers:%d head_num:%d gcn_dropout:%1.2f mhsa_dropout:%1.2f syn_srd:%d sem_srd:%d \n" \
                          $idx $lr ${bert_lr[i]} $l2reg $num_layers $head_num $gcn_dropout $mhsa_dropout $syn_srd $sem_srd

                        python -u train.py --model_name semmhsa --num_epoch 200 --DEVICE 0 --gcn_dropout $gcn_dropout --l2reg $l2reg --mhsa_dropout $mhsa_dropout \
                          --threshold $threshold --lr $lr --dataset Restaurants --hidden_dim 204 --rnn_hidden 204 --syn_srd $syn_srd --sem_srd $sem_srd \
                          --num_layers $num_layers --emb_type bert --log_step 40 --head_num $head_num --batch_size 32  \
                          > ./out_rest_bert/$idx.out 2>&1
                    fi

                    idx=`expr $idx + 1`
                    sem_srd=`expr $sem_srd + 1`
                  done

                  syn_srd=`expr $syn_srd + 1`
                done
                
                threshold=`echo "scale=2; $threshold + .10" | bc`  #`echo "$dropout + .10"|bc`
              done
            mhsa_dropout=`echo "scale=2; $mhsa_dropout + .10" | bc`  #`echo "$dropout + .10"|bc`
            done

            gcn_dropout=`echo "scale=2; $gcn_dropout + .10" | bc`  #`echo "$dropout + .10"|bc`
          done
          head_num=`expr $head_num + 1`

        done
        num_layers=`expr $num_layers + 1`

      done

    l2reg=`echo "scale=6; $l2reg / 100" | bc`
    done

    lr=`echo "scale=6; $lr / 10" | bc`
  done
done

echo "finish"
