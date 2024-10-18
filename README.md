### TrieLLM 


### Introduction [[知乎]](https://zhuanlan.zhihu.com/p/1604769906)
A simple example to control LLM for text generations via a Custom Trie (prefix tree).

The large language models (LLMs) generate complete sentences by selecting characters from the vocabulary one by one based on their probabilities.

For example, given 3 three sequences,

        <a_128><b_241><c_146><d_235>
        <a_171><b_57><c_141><d_231>
        <a_135><b_16><c_77><d_23>


the first four choices are fixed and sequential ('<' → 'a' → '_' → '1'), with the fifth choice restricted to {'2', '7', '3'}.

### Instruction

#### (1) Install requirements 

    >> pip install -r requirements.txt


#### (2) Apply for a granted access at [[Hugging Face]](https://huggingface.co/meta-llama/Llama-3.2-1B)

    >> huggingface-cli login --token [YOUR TOKEN]

#### (3) Run generate.py

    >> python generate.py



### ✨✨✨✨ 
- If this repository helps you, please star it. Thank you ~
- If you have any questions, please feel free to contact me at kaysenn@163.com.

