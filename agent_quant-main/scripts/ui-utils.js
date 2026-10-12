function animateNumber(number) {
    if (typeof number !== 'number') {
        throw new TypeError('Expected a number');
    }
    const strNumber = number.toString();
    // Existing animation code that possibly calls .replace()
    return strNumber.replace(/\/g, ''); // Example usage
}