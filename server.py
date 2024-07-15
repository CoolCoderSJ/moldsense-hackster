from flask import Flask, render_template
import requests
import torch
import time
import os
import psutil
import transformers
from transformers import AutoTokenizer, set_seed
import qlinear
import logging

set_seed(123)
transformers.logging.set_verbosity_error()
logging.disable(logging.CRITICAL)


app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/investigate')
def investigate():
    r = requests.get('http://192.168.40.199/collect').text
    
    prompt = """The following data was collected from a piece of bread confirmed as not expired:
temperature	gas	humidity	pressure	pH
24.80631579	72337.44085	40.21393421	983.6218421	4.061113759
24.80934211	17780.14345	40.19061842	983.6206579	4.159856247
24.81763158	19909.41885	40.31140789	983.6218421	3.511640852
24.82486842	21438.88135	40.54214474	983.6244737	3.45820374
24.82921053	22084.85617	40.81	983.6252632	3.494215707
24.83	22262.88773	41.30361842	983.6252632	3.501185765
24.82763158	22866.18845	41.74636842	983.6292105	3.230515178
24.82210526	23458.83729	42.07785526	983.6285526	2.936611065
24.81434211	23806.65114	42.32146053	983.6290789	3.482598943
24.80447368	24088.37537	42.53134211	983.6297368	2.963329621
24.79394737	24427.01862	42.72256579	983.63	2.914539215
24.78078947	24612.44632	42.92011842	983.6315789	3.031868525
24.76828947	24661.3156	43.15543421	983.6343421	3.381533102
24.75407895	25040.41571	43.32809211	983.6326316	3.224706797
24.73934211	25596.94426	43.39698684	983.6332895	3.103892457

The following data was collected from the same piece of bread after it expired:
temperature	gas	humidity	pressure	pH
23.78626667	392733.1893	44.50936	977.4632	9.80753937
24.20413333	390431.8406	47.06976	969.7296	10.18610505
24.2396	395840.9034	47.11681333	969.7277333	7.788700787
24.27786667	399261.78	47.23146667	969.7314667	7.616200787
24.31213333	401653.8812	47.42829333	969.7312	6.658622047
24.34213333	403358.9794	47.70202667	969.7324	6.430783592
24.36866667	404684.682	48.01273333	969.7302667	6.84433003
24.39173333	405779.1257	48.32637333	969.732	6.857899325
24.40986667	406493.3329	48.62576	969.736	7.294810676
24.42693333	406940.5063	48.9312	969.7341333	6.711042692
24.43973333	407265.8817	49.25348	969.7358667	6.066577119
24.4516	407553.8909	49.56748	969.7364	6.873681102
24.4624	407808.2898	49.85973333	969.7345333	6.453413492
24.47133333	408101.1169	50.12746667	969.736	7.095154199
24.47986667	408378.6175	50.36673333	969.7362667	7.170169639

Now, the following data was collected from another piece of bread:
temperature	gas	humidity	pressure	pH
23.05842105	84506.50206	40.03444737	963.0113158	5.064802114
23.06723684	21512.75121	39.98852632	963.0118421	5.039245234
23.08842105	23396.20081	40.16598684	963.0098684	4.91378419
23.11131579	23997.13353	40.73603947	963.0168421	5.532957677
23.13236842	24444.94926	41.45980263	963.0182895	4.871963842
23.15078947	24722.49941	42.17094737	963.0178947	4.932371011
23.16789474	24932.19173	42.80332895	963.0180263	4.614071695
23.18236842	25436.07395	43.29017105	963.0188158	4.119197576
23.19421053	25494.41482	43.66269737	963.0210526	4.035556879
23.20447368	25878.28353	43.97667105	963.0257895	4.214455035
23.21328947	26425.3669	44.18727632	963.0290789	4.485125622
23.21986842	26749.61537	44.34909211	963.0321053	4.637305222
23.22644737	26970.38431	44.48267105	963.0323684	4.789484822
23.23157895	27181.96829	44.55442105	963.0346053	5.025305118
23.23578947	27560.03526	44.57518421	963.0338158	4.695389038

Based on the data collected from the first piece of bread, estimate a percentage for how expired the second piece of bread is. Reply only with the percentage."""
    p = psutil.Process()
    p.cpu_affinity([0, 1, 2, 3])
    torch.set_num_threads(4)

    tokenizer = AutoTokenizer.from_pretrained("llama3-8b-amd-npu")
    ckpt = "llama3-8b-amd-npu/pytorch_llama3_8b_w_bit_4_awq_lm_amd.pt"
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    model = torch.load(ckpt)
    model.eval()
    model = model.to(torch.bfloat16)

    for n, m in model.named_modules():
        if isinstance(m, qlinear.QLinearPerGrp):
            print(f"Preparing weights of layer : {n}")
            m.device = "aie"
            m.quantize_weights()


    input = tokenizer.apply_chat_template(
    [{
        "role": "user",
        "content": prompt
    }],
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
    )

    outputs = model.generate(input['input_ids'],
    max_new_tokens=10,
        eos_token_id=terminators,
    attention_mask=input['attention_mask'],
        do_sample=True,
        temperature=0.6,
        top_p=0.9)

    response = outputs[0][input['input_ids'].shape[-1]:]
    response_message = tokenizer.decode(response, skip_special_tokens=True)
    
    percentage = float(response_message.split("%")[0].split(" ")[-1])

    return render_template('investigate.html', percentage=percentage)


app.run()