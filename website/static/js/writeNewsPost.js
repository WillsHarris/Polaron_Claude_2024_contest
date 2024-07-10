function writeNewsPost(tickers, scores, newsText, post_format='wrap', img_path=null, ticker_links=null) {

  var c = document.getElementById("myCanvas");
  var ctx = c.getContext("2d");
  
  // tickers and scores
  var score = scores[0];
  var ticker = tickers[0];
  var ticker2 = tickers.length > 1 ? tickers[1] : null;
  var score2 = scores.length > 1 ? scores[1] : null;

  var scores = [score]
  if (score2 != null && score2 != score) {
    scores.push(score2)
  }
  var j = 0;
  var sf = 19;

  var centerX = 90 * sf;
  var centerY = 90 * sf;

  if (post_format == 'below') { 
    var centerX = 105 * sf;
  }

  var radius = 80 * sf;
  var delta_r = radius/18;
  var lineWidth = radius/130;
  var numberOfArcs = 7; // Adjust the number of arcs as needed
  var elemGap = 180 * sf;

  // color gradients
  var arc_colors = ['green', '#61B061', '#AAD2AA', 'lightgray', '#F0B1B2', '#F55B5B', '#F50000']; 
  var arc_colors = ['#47732b', '#82a46b', '#bed6ac', 'lightgray', '#e7b1b1', '#d57575', '#c74848']; 
  //var arc_colors = ['#0015F5', '#61B0FF', '#AAD2FA', 'lightgray', '#F0B1B2','#F55B5B','#F50000'];
  var line_colors = ['#585858', 'gray', '#ABABAB']

  var gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
  gradient.addColorStop(0, 'white');
  gradient.addColorStop(0.75, 'white');
  gradient.addColorStop(1, "rgba(230, 230, 230, 0.05)");




  const lineHeight = radius/5;
  const maxWidth = c.width;
  const extraRoom = radius/5; // Extra room to the right
  const fontSize = 300;
  const font = 'L';
  const textPadH = 0.1 * radius;

  // place main text
  if (post_format == 'wrap') {
    placeTextAroundCemi(ctx, newsText, centerX, centerY, radius, lineHeight, maxWidth, extraRoom, fontSize, font);
  } else {
    placeTextBelowSemicircle(ctx, newsText, textPadH, centerY, 0*radius, lineHeight, maxWidth - 2*textPadH, fontSize, font)
  }

  // draw the bearometer
  drawGauge(ctx, centerX, centerY, radius, lineWidth, sf, arc_colors, numberOfArcs, score, score2, delta_r);


  function drawImageAndLine() {
    var img = new Image();
    img.src = img_path;
  
    img.onload = function() {
      var w = img.width;
      var h = img.height;
      var aspectRatio = w / h;
      var img_x = -(centerX - radius - (lineWidth * sf) / 2) + (c.width - (aspectRatio * radius)); // such that right edge is at right edge of canvas
      ctx.drawImage(img, img_x, centerY - radius, radius * aspectRatio, radius);
  
      // add vertical line to canvas
      var line_x = ((centerX + radius + (lineWidth * sf) / 2) + img_x) / 2.05;
      ctx.beginPath();
      ctx.moveTo(line_x, centerY - radius);
      ctx.lineTo(line_x, centerY);
      ctx.lineWidth = 1 * lineWidth;
      // make it dashed
      // ctx.setLineDash([5, 35]);
      ctx.strokeStyle = '#000000';
      ctx.stroke();
    };
  
    img.onerror = function() {
      console.error("Failed to load image");
      // Optionally, you can display a placeholder image or retry loading
    };
  }
  
  // Call the function to draw the image and line
  drawImageAndLine();


  function addLogos() {
    function drawLogo(logoA, logoB, x, y) {
      var w1 = logoA.width;
      var h1 = logoA.height;
      var aspectRatio1 = w1 / h1;
      var v_shift = radius / 6;
      
      if (logoB != null) {
        var w2 = logoB.width;
        var h2 = logoB.height;
      } else {
        var w2 = null;
        var h2 = null;
      }

      var res = get_scaled_widths(radius, v_shift, w1, h1, w2, h2);
      ctx.drawImage(logoA, res.x1, res.y1, res.w1, res.h1);
    }

    var logo1 = new Image();
    var logo2 = new Image();
    logo1.src = 'logos/' + ticker + '_bw.png';
    logo2.src = 'logos/' + ticker2 + '_bw.png';

    logo1.onload = function() {
      
      if (score2 != null) {
        drawLogo(logo2, logo1, x=centerX, y=centerY);
      }
      else {
        logo2 = null;
      }
      drawLogo(logo1, logo2, x=centerX, y=centerY);
    };

    logo2.onload = function() {
      drawLogo(logo2, logo1, x=centerX, y=centerY);
      drawLogo(logo1, logo2, x=centerX, y=centerY);
    };
  }


  addLogos();
}