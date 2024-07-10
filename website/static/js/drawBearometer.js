function drawGauge(ctx, centerX, centerY, radius, lineWidth, sf, arc_colors, numberOfArcs, score, score2, delta_r) {
    // Draw the semicircles for the gauge
    for (var i = 0; i < numberOfArcs; i++) {
        // Calculate start and end angles for each arc
        var endAngle = -i * (Math.PI / numberOfArcs) + 0.002;
        var startAngle = -(i + 1) * (Math.PI / numberOfArcs);
        // Draw the arc
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
        ctx.lineWidth = lineWidth * sf; // Set the thickness of each arc
        // Set the stroke style to the calculated RGB values
        ctx.strokeStyle = arc_colors[i];

        if (i == 0) {
            ctx.lineCap = "round";
        } else {
            ctx.lineCap = "butt";
        }

        ctx.stroke();
    }

    // Wedge for the score
    var i = 3 - score;
    var endAngle = -i * (Math.PI / numberOfArcs);
    var startAngle = -(i + 1) * (Math.PI / numberOfArcs);
    ctx.beginPath();
    ctx.moveTo(centerX, centerY); // Move to the center
    ctx.arc(centerX, centerY, radius - 1.75 * delta_r, startAngle, endAngle);
    ctx.closePath(); // Close the path to create a wedge
    ctx.fillStyle = arc_colors[i];
    ctx.globalAlpha = 0.8;
    ctx.fill();
    ctx.globalAlpha = 1.0;

    // Wedge for the score2 if it is different from score
    if (score2 != score && score2 != null) {
        var i = 3 - score2;
        var endAngle = -i * (Math.PI / numberOfArcs);
        var startAngle = -(i + 1) * (Math.PI / numberOfArcs);
        ctx.beginPath();
        ctx.moveTo(centerX, centerY); // Move to the center
        ctx.arc(centerX, centerY, radius - 1.75 * delta_r, startAngle, endAngle);
        ctx.closePath(); // Close the path to create a wedge
        ctx.fillStyle = arc_colors[i];
        ctx.globalAlpha = 0.8;
        ctx.fill();
        ctx.globalAlpha = 1.0;
    }

    // Shading for wedge
    for (var i = 0; i < 60; i++) {
        // Calculate start and end angles for each arc
        var endAngle = 0;
        var startAngle = -Math.PI;
        // Draw the arc
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius - (i / 4 + 2) * delta_r, startAngle, endAngle);
        ctx.closePath();
        ctx.lineWidth = lineWidth * sf; // Set the thickness of each arc
        // Set the stroke style to the calculated RGB values
        ctx.fillStyle = 'white';
        ctx.globalAlpha = 0.03;
        ctx.fill();
    }
    ctx.globalAlpha = 1;

    // fill left side of the arc (hack to get round end cap on one end only)
    var i = numberOfArcs-1; // the last arc (red one)
    ctx.beginPath();
    var endAngle = -i*(Math.PI/numberOfArcs)-0.4;
    var startAngle = -(i+1)*(Math.PI/numberOfArcs);
    ctx.arc(centerX, centerY, radius, startAngle, endAngle);
    ctx.lineWidth = lineWidth * sf; // Set the thickness of each arc
    ctx.strokeStyle = arc_colors[i];
    ctx.lineCap = "round";
    ctx.stroke();


    // "+" and "-" signs
    var i = numberOfArcs;
    var midAngle = endAngle = -(i-0)* (Math.PI / numberOfArcs);
    var lineEndX = centerX + radius * Math.cos(midAngle);
    var lineEndY = centerY + radius * Math.sin(midAngle);
    var x = lineEndX; 
    var y = lineEndY;

    ctx.font = '560px Helvetica';
    ctx.fillStyle = 'black';
    ctx.textAlign = 'center';      // Align text horizontally center
    ctx.textBaseline = 'middle';   // Align text vertically middle
    ctx.fillText('-', x+ sf/10, y+sf/10);

    var i = 0;
    var midAngle = endAngle = -(i-0)* (Math.PI / numberOfArcs);
    var lineEndX = centerX + radius * Math.cos(-midAngle);
    var lineEndY = centerY + radius * Math.sin(-midAngle);
    var x = lineEndX;
    var y = lineEndY;

    ctx.font = 'bold 300px Helvetica';
    ctx.fillStyle = 'black';
    ctx.textAlign = 'center';      // Align text horizontally center
    ctx.textBaseline = 'middle';   // Align text vertically middle
    ctx.fillText('+', x- sf/10, y- sf/10);
}
