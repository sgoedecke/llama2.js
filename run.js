const { rmsnorm, softmax, randomSample, matmul, colours, splitFloat32Array, pauseForKeypress} = require('./utils.js')
const { loadCheckpoint } = require('./loadCheckpoint.js');
const { loadTokenizer, tokenizePrompt } = require('./tokenizer.js');
const { parseArgs } = require('node:util') // need a recent version of node

const args = ['--num', '--prompt'];
const cliOptions = {
  num: {
    type: 'string',
    short: 'n',
  },
  prompt: {
    type: 'string',
    short: 'p'
  },
  emphasis: { // trying something out; manually overriding att
    type: 'string',
    short: 'e'
  },
};
const cliOptionValues = parseArgs({ args: process.argv, options: cliOptions, allowPositionals: true }).values


const checkpoint_path = "stories42M.bin" // 42, 15
const tokenizer_path = "llama2c/tokenizer.bin"
const prompt = cliOptionValues.prompt || "Once upon a time, there was a "
const steps = Number(cliOptionValues.num) // the -n param in llama2.c

const emphasis = cliOptionValues.emphasis

const { weights, config, headSize } = loadCheckpoint(checkpoint_path)
const tokenizer = loadTokenizer(tokenizer_path, config) // need vocabSize from config due to peculiarities of llama2 bin format
const promptTokens = tokenizePrompt(prompt, tokenizer)


debugger

// Now we have our prompt tokenized into a list of token ids
// Let's get started

let pos = 0
promptTokens.unshift(1) // 1 = BOS (beginning of string) token in Llama-2 sentence-piece
const numPromptTokens = promptTokens.length
let token  = promptTokens[0]
let next

// calculate position of emphasized string
const emphasisPos = { left: 0, right: 0 }
for (let i = 0; i < promptTokens.length; i++) {
    const head = promptTokens.slice(i).map((token) => tokenizer.vocab[token]).join("")
    if (!head.includes(emphasis)) {
        emphasisPos.left = i -1
        break
    }
}
for (let i = promptTokens.length; i>0; i--) {
    const tail = promptTokens.slice(-i).map((token) => tokenizer.vocab[token]).join("")
    if (!tail.includes(emphasis)) {
        emphasisPos.right = promptTokens.length - i + 1
        break
    }
}

const runState = {
    // current wave of activations
    x: new Float32Array(config.dim), // activation at current time stamp (dim,)
    xb: new Float32Array(config.dim), // same, but inside a residual branch (dim,)
    xb2: new Float32Array(config.dim), // an additional buffer just for convenience (dim,)
    hb: new Float32Array(config.hiddenDim), // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: new Float32Array(config.hiddenDim), // buffer for hidden dimension in the ffn (hidden_dim,)
    q: new Float32Array(config.dim), // query (dim,)
    k: new Float32Array(config.dim), // key (dim,)
    v: new Float32Array(config.dim), // value (dim,)
    // NB: this was a bug in at least one of the ports, I think: att was initialized too small (without * nHeads)
    att: new Float32Array(config.seqLen * config.nHeads), // buffer for scores/attention values // run.js has (n_heads, seq_len) instead
    logits: new Float32Array(config.vocabSize), // output logits
    // kv cache
    keyCache: new Float32Array(config.nLayers * config.seqLen * config.dim),   // (layer, sq_len, dim)
    valueCache: new Float32Array(config.nLayers * config.seqLen * config.dim), // (layer, seq_len, dim)
};
runState.keyCache = splitFloat32Array(runState.keyCache, config.nLayers)
    .map((arr) => splitFloat32Array(arr, config.seqLen)) // 3d tensor
runState.valueCache = splitFloat32Array(runState.valueCache, config.nLayers)
    .map((arr) => splitFloat32Array(arr, config.seqLen)) // 3d tensor

function nilOut(arr) {
    for (let i = 0; i < arr.length; i++) {
        arr[i] = 0
    }
}

const output = []

while (pos < steps) {
    // forward pass
    // nil out state in between passes to prove we can
    nilOut(runState.x)
    nilOut(runState.q)
    nilOut(runState.k)
    nilOut(runState.v)
    nilOut(runState.att)
    nilOut(runState.hb)
    nilOut(runState.hb2)
    nilOut(runState.xb)
    nilOut(runState.xb2)


    // copy the token embedding into x
    runState.x = new Float32Array(weights.tokenEmbeddingTable[token])

   
    // forward each layer
    for (let layer = 0; layer < config.nLayers; layer++) {
        // nil out state in between layers to prove we can
        nilOut(runState.q)
        nilOut(runState.k)
        nilOut(runState.v)
        nilOut(runState.att)
        nilOut(runState.hb)
        nilOut(runState.hb2)
        nilOut(runState.xb)
        nilOut(runState.xb2)
    
        rmsnorm(runState.xb, runState.x, weights.rmsAttWeight[layer]) // attention rmsnorm

        // attention qkv matmuls
        matmul(runState.q, runState.xb, weights.wq[layer])
        matmul(runState.k, runState.xb, weights.wk[layer])
        matmul(runState.v, runState.xb, weights.wv[layer])

        // RoPE relative pos encoding: complex-valued rotate q and k in each head
        // llama 2 doesn't use freqx from the weights, but the port I copied does
        for (let head = 0; head < config.nHeads; head++) {
            const start = head * headSize;
            for (let i = 0; i < headSize; i += 2) {
                const q0 = runState.q[start + i];  
                const q1 = runState.q[start + i + 1];
                const k0 = runState.k[start + i];
                const k1 = runState.k[start + i + 1];
                const fcr = weights.freqCisReal[pos][i / 2]
                const fci = weights.freqCisImag[pos][i / 2]

                runState.q[start + i] = q0 * fcr - q1 * fci;
                runState.q[start + i + 1] = q0 * fci + q1 * fcr;
                runState.k[start + i] = k0 * fcr - k1 * fci;
                runState.k[start + i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // populate key/value caches. llama.c does this way above here but all ports do this here
        runState.keyCache[layer][pos] = new Float32Array(runState.k)
        runState.valueCache[layer][pos] = new Float32Array(runState.v)

        // multi-head attention, iterate over all heads. could parallelize this in theory.
        for (let head = 0; head < config.nHeads; head++) {
            const att = runState.att.subarray(head * config.seqLen, (head + 1) * config.seqLen)          
            for (let t = 0; t <= pos; t++) {              
                // att score
                let score = 0;
                for (let i = 0; i < headSize; i++) {
                    // hmm
                    const qkDotProd = runState.q[head * headSize + i] * runState.keyCache[layer][t][head * headSize + i];
                    score += qkDotProd
                }

                score /= Math.sqrt(headSize);

                // save to att buffer  
                att[t] = score;
            }

            // juice the attention weights of any tokens in the emphasis segment
            for (let j = 0; j <= att.length; j++) {              
                if ((j >= emphasisPos.left - 1) && (j < emphasisPos.right -1)) {
                    // randomly-ish chosen emphasis factor. more seems to provide gibberish, less gets drowned out
                    att[j] += 2.5; 
                }
            }

            // softmax att weights  
            softmax(att, pos + 1);

            // weighted sum of v into xb
            for (let i = 0; i < headSize; i++) {
                runState.xb[head * headSize + i] = 0;
                for (let t = 0; t <= pos; t++) {
                    runState.xb[head * headSize + i] += att[t] * runState.valueCache[layer][t][head * headSize + i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(runState.xb2, runState.xb, weights.wo[layer])

        // residual connection 
        for (let i = 0; i < runState.xb2.length; i++) { // should be config.dim
            runState.x[i] += runState.xb2[i]; 
        }

        // Attention is done, now to do the feed-forward-network part
        rmsnorm(runState.xb, runState.x, weights.rmsFFNWeight[layer]) // ffn rmsnorm
        
        matmul(runState.hb, runState.xb, weights.w1[layer])
        matmul(runState.hb2, runState.xb, weights.w3[layer])

        // SwiGLU non-linearity (Gated Linear Unit)
        for (let i = 0; i < config.hiddenDim; i++) {
            // sigmoid - compresses input into a range between 0 and 1
            runState.hb[i] = runState.hb[i] * (1.0 / (1.0 + Math.exp(-runState.hb[i])));
        }
        // elementwise mult with w3(x) to apply the non-linear gating
        for (let i = 0; i < config.hiddenDim; i++) {
            runState.hb[i] = runState.hb[i] * runState.hb2[i];
        }

        // final matmul for ffn output
        matmul(runState.xb, runState.hb, weights.w2[layer])

        // residual connection
        for (let i = 0; i < runState.xb.length; i++) {
            runState.x[i] += runState.xb[i];
        }
    }
    // skip the logits step entirely if we're still in the prompt
    if (pos >= numPromptTokens-1) {
        // final rmsnorm
        rmsnorm(runState.x, runState.x, weights.rmsFinalWeight);

        // classifier into logits
        matmul(runState.logits, runState.x, weights.wcls); // could I just reuse tokenEmbeddingTable? some ports do
    }

    if (pos < numPromptTokens - 1) {
        // still in prompt
        next = promptTokens[pos + 1]
    } else {
        // simplest possible topp over the logits
        const numToSample = 1
        const topp = Array.from(runState.logits).sort((a, b) => b - a).slice(0, numToSample).map(x => runState.logits.indexOf(x))//.map(x => tokenizer.vocab[x])
        next = randomSample(topp)
    }

    let piece = tokenizer.vocab[next]
    if (piece == undefined) {
        piece = "<oops>"
        token = 1
    }

    // I should really decode these bytes instead, but it doesn't seem worth it
    if (piece.match(/^<0x/)) { piece = " " }

    // calculate the attention matrix at this token
    let mashedAtt = runState.att.subarray(0, config.seqLen)
    for (let head = 1; head < config.nHeads; head++) {
        const att = runState.att.subarray(head * config.seqLen, (head + 1) * config.seqLen)
        for (let i = 0; i < att.length; i++) {
            mashedAtt[i] += att[i]
        }
    }

    // calculate how much the top logits differ
    const topLogits = Array.from(runState.logits).sort((a, b) => b - a).slice(0, 3)
    const topLogitDiff = topLogits[0] - topLogits[2]

    output.push({
        piece: piece,
        token: token,
        att: mashedAtt,
        topLogitDiff: topLogitDiff
    })

    printWithHighlightedConfidence(output)
    // printWithHighlightedAttention(output)
    // printOutput(output)

    token = next

    pos += 1
}

// calculate and print a summary of the total highlighted confidence
averageTLD = output.map(c => c.topLogitDiff).reduce((a, b) => a + b, 0) / output.length
averageTLD *= 100 // scale for easier reading
averageTLD = averageTLD.toFixed(2)
console.log("\nGeneration completed. Confidence level: " + (averageTLD / output.length))

function printWithHighlightedConfidence(output) {
    const lastChunk = output[output.length - 1]
    let colour
    let buf = ''
    const diff = lastChunk.topLogitDiff
    if (diff > 6) {
        colour = colours.bright
    } else if (diff > 2) {
        colour = colours.reset
    } else {
        colour = colours.dim
    }
    buf += colour
    buf += lastChunk.piece
    buf += colours.reset
    process.stdout.write(buf)
}

function printOutput(output) {
    process.stdout.write(output[output.length - 1].piece)
}

function printWithHighlightedAttention(output) {
    console.clear() // need to clear and rewrite each time because previous att changes

    buf = ''
    output.forEach((chunk, i) => {
        let colour
        const currentAttention = output[output.length - 1].att[i]
        if (currentAttention > 0.7) {
            colour = colours.bright
        } else if (currentAttention > 0.1) {
            colour = colours.reset
        } else {
            colour = colours.dim
        }
        buf += colour
        buf += chunk.piece
        buf += colours.reset
    })
    process.stdout.write(buf)
}
