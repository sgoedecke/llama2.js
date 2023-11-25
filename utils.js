const fs = require('fs');

function matmul(output, input, weight) {
    for (let i = 0; i < output.length; i++) {
        output[i] = 0;
        for (let j = 0; j < input.length; j++) {
            output[i] += input[j] * weight[i][j];
        }
    }
}

function rmsnorm(o, x, weight) {
    const size = x.length
    // calculate sum of squares  
    let ss = 0.0;
    for (let j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5; 
    ss = 1.0 / Math.sqrt(ss);
    
    // normalize and scale
    for (let j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

function softmax(x, size) {   
    // find max value 
    let maxVal = x[0];
    for (let i = 1; i < size; i++) {
        if (x[i] > maxVal) {
            maxVal = x[i]; 
        }
    }
    
    // exp and sum
    let sum = 0;
    for (let i = 0; i < size; i++) {
        x[i] = Math.exp(x[i] - maxVal);
        sum += x[i];
    }
    
    // normalize 
    for (let i = 0; i < size; i++) {
        x[i] /= sum;
    }
    return x
    
}

function randomSample(array) {
    const index = Math.floor(Math.random() * array.length);
    return array[index];
}

const colours = {
    reset: "\x1b[0m",
    bright: "\x1b[1m",
    dim: "\x1b[2m",
    underscore: "\x1b[4m",
    blink: "\x1b[5m",
    reverse: "\x1b[7m",
    hidden: "\x1b[8m",
    
    fg: {
        black: "\x1b[30m",
        red: "\x1b[31m",
        green: "\x1b[32m",
        yellow: "\x1b[33m",
        blue: "\x1b[34m",
        magenta: "\x1b[35m",
        cyan: "\x1b[36m",
        white: "\x1b[37m",
        gray: "\x1b[90m",
        crimson: "\x1b[38m" // Scarlet
    },
    bg: {
        black: "\x1b[40m",
        red: "\x1b[41m",
        green: "\x1b[42m",
        yellow: "\x1b[43m",
        blue: "\x1b[44m",
        magenta: "\x1b[45m",
        cyan: "\x1b[46m",
        white: "\x1b[47m",
        gray: "\x1b[100m",
        crimson: "\x1b[48m"
    }
};

function splitFloat32Array(array, numChunks) {
    const result = [];

    const chunkSize = array.length / numChunks
    for (let i = 0; i < array.length; i += chunkSize) {
        const end = Math.min(i + chunkSize, array.length);
        result.push(array.subarray(i, end));
    }
    return result;
}

function pauseForKeypress() {
    var fd = fs.openSync("/dev/stdin", "rs")
    fs.readSync(fd, new Buffer(1), 0, 1)
    fs.closeSync(fd)
}


module.exports = { softmax, randomSample, matmul, rmsnorm, colours, splitFloat32Array, pauseForKeypress }
