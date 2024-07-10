// semicircleText.js

function drawSemicircle(ctx, centerX, centerY, radius) {
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI, true);
    ctx.closePath();
    ctx.stroke();
}

function placeTextAroundCemi(ctx, text, centerX, centerY, radius, lineHeight, maxWidth, extraRoom, fontSize, font='Lexend') {
    const words = text.split(' ');
    let line = '';
    let y = centerY - radius + lineHeight; // Start a bit below the top of the semicircle

    ctx.font = fontSize + 'px ' + font;
    ctx.textBaseline = 'bottom';

    for (let i = 0; i < words.length; i++) {
        const testLine = line + words[i] + ' ';
        const metrics = ctx.measureText(testLine);
        const testWidth = metrics.width;

        const x = centerX + Math.sqrt(radius * radius - Math.pow(centerY - y, 2)) + extraRoom;

        if (testWidth > (maxWidth - x - 100) && i > 0) {
            ctx.fillText(line, x, y);
            line = words[i] + ' ';
            y += lineHeight;

            // Ensure y does not exceed the bottom of the semicircle
            if (y > centerY) {
                break;
            }
        } else {
            line = testLine;
        }
    }

    // Final line
    if (y <= centerY) {
        const x = centerX + Math.sqrt(radius * radius - Math.pow(centerY - y, 2)) + extraRoom;
        ctx.fillText(line, x, y);
    }
}
function placeTextBelowSemicircle(ctx, text, centerX, centerY, radius, lineHeight, maxWidth, fontSize, justify = true, font = 'Lexend') {
    //var centerX = 26*centerX
    const words = text.split(' ');
    let line = '';
    let lines = [];
    let y = centerY + radius + lineHeight; // Start below the bottom of the semicircle

    ctx.font = `${fontSize}px ${font}`;
    ctx.textBaseline = 'top';

    for (let i = 0; i < words.length; i++) {
        const testLine = line + words[i] + ' ';
        const metrics = ctx.measureText(testLine);
        const testWidth = metrics.width;

        if (testWidth > maxWidth && i > 0) {
            lines.push(line.trim());
            line = words[i] + ' ';
            y += lineHeight;

            // Check if the new line exceeds the canvas height
            if (y > ctx.canvas.height) {
                console.warn('Text exceeds canvas height, some text may not be visible.');
                break;
            }
        } else {
            line = testLine;
        }
    }

    // Add the final line
    lines.push(line.trim());

    // Draw the lines
    y = centerY + radius + lineHeight; // Reset y position
    for (let i = 0; i < lines.length; i++) {
        if (justify && i < lines.length - 1) {
            drawJustifiedLine(ctx, lines[i], centerX, y, maxWidth);
        } else {
            const x = centerX; // Center align
            ctx.fillText(lines[i], x, y);
        }
        y += lineHeight;
    }
}

function drawJustifiedLine(ctx, line, centerX, y, maxWidth) {
    const words = line.split(' ');
    const totalWords = words.length;
    if (totalWords === 1) {
        ctx.fillText(line, centerX, y);
        return;
    }

    let lineWidth = ctx.measureText(line.replace(/\s+/g, '')).width; // width without spaces
    let spaceWidth = (maxWidth - lineWidth) / (totalWords - 1);

    let x = centerX; // Start at the left edge
    for (let i = 0; i < totalWords; i++) {
        ctx.fillText(words[i], x, y);
        x += ctx.measureText(words[i]).width + spaceWidth;
    }
}
