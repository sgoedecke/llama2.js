const fs = require("fs")
const FLOAT_SIZE = 4

function loadTokenizer(tokenizer_path, config) {
    // OK, now let's build the tokenizer
    console.log("Loading model checkpoint from", tokenizer_path)
    const tokenizerBin = fs.readFileSync(tokenizer_path)

    const tokenizer = {}
    tokenizer.maxTokenLength = tokenizerBin.readUInt32LE(0)
    offset = FLOAT_SIZE
    // we're skipping byte pieces and sortedVocab as unnecessary for now

    tokenizer.vocabScores = []
    tokenizer.vocab = []
    for (let i = 0; i < config.vocabSize; i++) {
        score = tokenizerBin.readFloatLE(offset)
        offset += FLOAT_SIZE
        tokenizer.vocabScores[i] = score
        
        len = tokenizerBin.readInt32LE(offset)
        offset += FLOAT_SIZE
        
        let str = "";
        for (let j = 0; j < len; j++) {
            str += String.fromCharCode(tokenizerBin[offset++]);
        }
        tokenizer.vocab[i] = str;
    }
    return tokenizer
}

function tokenizePrompt(prompt, tokenizer) {
    if (prompt[0] != " ") { prompt = " " + prompt } // see add_dummy_prefix from llama2
    const promptTokens = []
    
    prompt.split("").forEach((char) => {
        promptTokens.push(tokenizer.vocab.indexOf(char))
    })
    
    const tries = 50
    for (let i = 0; i < tries; i++) {
        let bestScore = -1e10
        let bestId = -1
        let bestIdx = -1
    
        for (let i = 0; i < promptTokens.length - 1; ++i) {
            const token = promptTokens[i]
            const nextToken = promptTokens[i+1]
            if (nextToken == undefined) {
                continue
            }
            const mergedToken = tokenizer.vocab[token] + tokenizer.vocab[nextToken]
            const mergedTokenIndex = tokenizer.vocab.indexOf(mergedToken)
            
            if (mergedTokenIndex == undefined || mergedTokenIndex == -1) {
                continue
            }

            const mergedTokenScore = tokenizer.vocabScores[mergedTokenIndex]           
            if (mergedTokenScore > bestScore) {
                bestScore = mergedTokenScore
                bestId = mergedTokenIndex
                bestIdx = i
            }
        }
    
        if (bestIdx == -1) { break } // no mergeable pairs left, we're done
    
        promptTokens.splice(bestIdx, 2, bestId)
    }
    return promptTokens
}

module.exports = { loadTokenizer, tokenizePrompt }

