
class Util:

    def overlay(imgBack, imgFront):

        hf, wf, _ = imgFront.shape
        hb, wb, _ = imgBack.shape

        x1, y1 = 0,0
        x2, y2 = min(wf, wb), min(hf, hb)

        # For negative positions, change the starting position in the overlay image
        x1_overlay = 0 
        y1_overlay = 0 

        # Calculate the dimensions of the slice to overlay
        wf, hf = x2 - x1, y2 - y1

        # If overlay is completely outside background, return original background
        if wf <= 0 or hf <= 0:
            return imgBack

        # Extract the alpha channel from the foreground and create the inverse mask
        alpha = imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 3] / 255.0
        inv_alpha = 1.0 - alpha

        # Extract the RGB channels from the foreground
        imgRGB = imgFront[y1_overlay:y1_overlay + hf, x1_overlay:x1_overlay + wf, 0:3]

        # Alpha blend the foreground and background
        for c in range(0, 3):
            imgBack[y1:y2, x1:x2, c] = imgBack[y1:y2, x1:x2, c] * inv_alpha + imgRGB[:, :, c] * alpha

        return imgBack

