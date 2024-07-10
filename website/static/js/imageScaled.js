function get_scaled_widths(radius, v_shift, w1, h1, w2=null, h2=null) {
    var r_scale = 1.15;
    var aspectRatio1 = w1 / h1;
    var img_gap = radius / 8;
  
    if (w2 != null && h2 != null) {
      // scale the images to have the same area
      var aspectRatio2 = w2 / h2;
      var area1 = w1 * h1;
      var scale = Math.sqrt(area1 / (w2 * h2));
      w2 = w2 * scale;
      h2 = h2 * scale;
  
      if (aspectRatio2 > 1.3 | aspectRatio1 > 1.3) {
  
        // Equation 2: W = (w2 * sf) / 2
        // Equation 3: H = (h1 * sf + h2 * sf + img_gap)
  
        // Substitute W and H in Equation 1: W^2 + H^2 = (r / r_scale)^2
        // ((w2 * sf) / 2)^2 + (h1 * sf + h2 * sf + img_gap)^2 = (r / r_scale)^2
  
        const rightSide = (radius / r_scale) ** 2;
  
        // Isolate sf in the equation
        // (w2^2 * sf^2) / 4 + (h1 * sf + h2 * sf + img_gap)^2 = (r / r_scale)^2
        // Define a quadratic equation in the form: a * sf^2 + b * sf + c = 0
       
        // w_top is w1 or w2 depending on which aspect ratio is smaller
        var w_top = w1;
        if (aspectRatio2 < aspectRatio1) {
          w_top = w2;
        }
        const a = (w_top ** 2) / 4 + (h1 + h2) ** 2;
        const b = 2 * (h1 + h2) * img_gap;
        const c = img_gap ** 2 - rightSide;
  
        // Use the quadratic formula to solve for sf: sf = (-b ± sqrt(b^2 - 4ac)) / 2a
        const discriminant = b ** 2 - 4 * a * c;
  
        if (discriminant < 0) {
            // No real solutions
            return null;
        }
  
        const sqrtDiscriminant = Math.sqrt(discriminant);
        const sf1 = (-b + sqrtDiscriminant) / (2 * a);
        const sf2 = (-b - sqrtDiscriminant) / (2 * a);
  
      
        if (sf1 > 0) {
          sf = sf1;
        } else if (sf2 > 0) {
          sf = sf2;
        } else {
          return null;
        }
  
        x  = x - w1 * sf / 2;
        y = y - h1 * sf 
        if (aspectRatio1 < aspectRatio2) {
          y = y - img_gap - (h2 * sf);
        }
      } 
      else { 
        // place logos side-by-side
        // Equation 2: W = ((w1 + w2) * sf  + img_gap)/2
        // Equation 3: H = max(h1, h2) * sf + v_shift
        // Substitute W and H in Equation 1: W^2 + H^2 = (r / r_scale)^2
        // solve for sf
        v_shift = v_shift/2;
        const maxHeight = Math.max(h1, h2);
        const rightSide = (radius / r_scale) ** 2;
        const a = ((w1 + w2) ** 2) / 4 + maxHeight ** 2;
        const b = (w1 + w2) * img_gap / 2 + maxHeight * v_shift;;
        const c = (img_gap ** 2) / 4 - rightSide + v_shift ** 2;
        const discriminant = b ** 2 - 4 * a * c;
        if (discriminant < 0) {
            // No real solutions
            return null;
        }
        const sqrtDiscriminant = Math.sqrt(discriminant);
        const sf1 = (-b + sqrtDiscriminant) / (2 * a);
        const sf2 = (-b - sqrtDiscriminant) / (2 * a);
  
        if (sf1 > 0) {
            sf = sf1;
          } else if (sf2 > 0) {
            sf = sf2;
          } else {
            console.log('no real solutions')
            return null;
          }
        if (w1 > w2) {
          x = x - w1 * sf / 2  - w2 * sf / 2 - img_gap / 2;
        }
        else {
          x = x - w1 * sf / 2  + w2 * sf / 2 + img_gap / 2;
        }
        y = y - h1 * sf - v_shift;
      }
    } else if (aspectRatio1 < 1.3) {
      var currentRadius = Math.sqrt((w1 * w1) / 4 + h1 * h1);
      var sf = radius / currentRadius / r_scale;
      // shift x and y to center the image
      x  = x - w1 * sf / 2;
      if (aspectRatio1 > 1.3) {
      y = y - h1 * sf - v_shift 
      } else {
      y = y - h1 * sf 
      };
    } else {
      var A = (w1 * w1) / 4 + h1 * h1;
      var B = 2 * h1 * v_shift;
      var C = v_shift * v_shift - radius * radius;
      var discriminant = B * B - 4 * A * C;
      if (discriminant < 0) {
        return null; // No real solution
      }
      var sf1 = (-B + Math.sqrt(discriminant)) / (2 * A);
      var sf2 = (-B - Math.sqrt(discriminant)) / (2 * A);
      var sf = Math.max(sf1, sf2) / r_scale;
      // shift x and y to center the image
      x  = x - w1 * sf / 2;
      if (aspectRatio1 > 1.3) {
      y = y - h1 * sf - v_shift 
      } else {
      y = y - h1 * sf
      };
  }
  return { w1: Math.round(w1 * sf), 
           h1: Math.round(h1 * sf), 
           x1: Math.round(x),
           y1: Math.round(y),
           w2: null, 
           h2: null,
           x2: null,
           y2: null};
  }