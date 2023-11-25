const fs = require('fs');
const { splitFloat32Array } = require('./utils.js')

function loadCheckpoint(checkpoint_path) {
    const FLOAT_SIZE = 4

    const checkpoint = fs.readFileSync(checkpoint_path)
    const configKeys  = ['dim',
    'hiddenDim',
    'nLayers',
    'nHeads',
    'nKVHeads',
    'vocabSize',
    'seqLen']
    const config = {}
    configKeys.forEach((key, i) => {
        config[key] = checkpoint.readUInt32LE(i * FLOAT_SIZE) // size of float
    })

    console.log(config)

    offset = configKeys.length * FLOAT_SIZE

    const weights = {}

    const headSize = config.dim / config.nHeads;

    weights.tokenEmbeddingTable = new Float32Array(checkpoint.buffer, offset, config.vocabSize * config.dim);
    offset += weights.tokenEmbeddingTable.byteLength;

    weights.rmsAttWeight = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim);
    offset += weights.rmsAttWeight.byteLength;

    weights.wq = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim * config.nHeads * headSize);
    offset += weights.wq.byteLength;

    weights.wk = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim * config.nKVHeads * headSize);
    offset += weights.wk.byteLength;

    weights.wv = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim * config.nKVHeads * headSize);
    offset += weights.wv.byteLength;

    weights.wo = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.nHeads * headSize * config.dim);
    offset += weights.wo.byteLength;

    weights.rmsFFNWeight = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim);
    offset += weights.rmsFFNWeight.byteLength;

    weights.w1 = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim * config.hiddenDim);
    offset += weights.w1.byteLength;

    weights.w2 = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.hiddenDim * config.dim );
    offset += weights.w2.byteLength;

    weights.w3 = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim * config.hiddenDim);
    offset += weights.w3.byteLength;

    weights.rmsFinalWeight = new Float32Array(checkpoint.buffer, offset, config.dim);
    offset += weights.rmsFinalWeight.byteLength;

    // this is temporary - seeing if the cpp style RoPE works
    weights.freqCisReal = new Float32Array(checkpoint.buffer, offset, config.seqLen * headSize / 2);
    offset += weights.freqCisReal.byteLength;
    weights.freqCisImag = new Float32Array(checkpoint.buffer, offset, config.seqLen * headSize / 2);
    offset += weights.freqCisImag.byteLength;

    // offset += config.seqLen * headSize / 2; // run.cpp reads in freq_cis_real from here but run.c now skips it
    // offset += config.seqLen * headSize / 2;

    weights.wcls = weights.tokenEmbeddingTable; // no shared weights

    // We've read out all the content from the checkpoint, but we're not in C anymore and we don't have to pretend that our
    // 2D arrays are represented by a 1D array. Let's fix that.


    // Let's resize the 1d weights arrays into 2d and 3d tensors
    weights.tokenEmbeddingTable = splitFloat32Array(weights.tokenEmbeddingTable, config.vocabSize); // [vocabSize, dim]
    weights.rmsAttWeight = splitFloat32Array(weights.rmsAttWeight, config.nLayers); // [layers, dim]
    weights.rmsFFNWeight = splitFloat32Array(weights.rmsFFNWeight, config.nLayers); // [layers, dim]

    // All these are attention matmuls with the 3d tensor format [layers, [dim, dim]]
    weights.wq = splitFloat32Array(weights.wq, config.nLayers)
        .map((array) => splitFloat32Array(array, config.dim)) // [layers, [dim, dim]]
    weights.wk = splitFloat32Array(weights.wk, config.nLayers)
        .map((array) => splitFloat32Array(array, config.dim)); // [layers, [dim, dim]]
    weights.wv = splitFloat32Array(weights.wv, config.nLayers)
        .map((array) => splitFloat32Array(array, config.dim));
    weights.wo = splitFloat32Array(weights.wo, config.nLayers)
        .map((array) => splitFloat32Array(array, config.dim));

    // These are the FFN (feed-forward-network) matmuls with the 3d tensor format [layers, [dim, hiddenDim]]
    weights.w1 = splitFloat32Array(weights.w1, config.nLayers)
        .map((array) => splitFloat32Array(array, config.hiddenDim));
    weights.w2 = splitFloat32Array(weights.w2, config.nLayers) // w2 is [layers, [hiddenDim, dim]]
        .map((array) => splitFloat32Array(array, config.dim));
    weights.w3 = splitFloat32Array(weights.w3, config.nLayers)
        .map((array) => splitFloat32Array(array, config.hiddenDim));

    weights.freqCisReal = splitFloat32Array(weights.freqCisReal, config.seqLen);
    weights.freqCisImag = splitFloat32Array(weights.freqCisImag, config.seqLen);


    weights.wcls = weights.tokenEmbeddingTable

    return { weights, config, headSize }
}

module.exports = { loadCheckpoint }