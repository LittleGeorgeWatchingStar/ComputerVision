from PIL import Image, ImageDraw
import numpy as np
import csv
import math

def ReadKeys(image):
    """Input an image and its associated SIFT keypoints.

    The argument image is the image file name (without an extension).
    The image is read from the PGM format file image.pgm and the
    keypoints are read from the file image.key.

    ReadKeys returns the following 3 arguments:

    image: the image (in PIL 'RGB' format)

    keypoints: K-by-4 array, in which each row has the 4 values specifying
    a keypoint (row, column, scale, orientation).  The orientation
    is in the range [-PI, PI] radians.

    descriptors: a K-by-128 array, where each row gives a descriptor
    for one of the K keypoints.  The descriptor is a 1D array of 128
    values with unit length.
    """
    im = Image.open(image+'.pgm').convert('RGB')
    keypoints = []
    descriptors = []
    first = True
    with open(image+'.key','r') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC,skipinitialspace = True)
        descriptor = []
        for row in reader:
            if len(row) == 2:
                assert first, "Invalid keypoint file header."
                assert row[1] == 128, "Invalid keypoint descriptor length in header (should be 128)."
                count = row[0]
                first = False
            if len(row) == 4:
                keypoints.append(np.array(row))
            if len(row) == 20:
                descriptor += row
            if len(row) == 8:
                descriptor += row
                assert len(descriptor) == 128, "Keypoint descriptor length invalid (should be 128)."
                #normalize the key to unit length
                descriptor = np.array(descriptor)
                descriptor = descriptor / math.sqrt(np.sum(np.power(descriptor,2)))
                descriptors.append(descriptor)
                descriptor = []
    assert len(keypoints) == count, "Incorrect total number of keypoints read."
    print ("Number of keypoints read:"), int(count)
    return [im,keypoints,descriptors]

def AppendImages(im1, im2):
    """Create a new image that appends two images side-by-side.

    The arguments, im1 and im2, are PIL images of type RGB
    """
    im1cols, im1rows = im1.size
    im2cols, im2rows = im2.size
    im3 = Image.new('RGB', (im1cols+im2cols, max(im1rows,im2rows)))
    im3.paste(im1,(0,0))
    im3.paste(im2,(im1cols,0))
    return im3

def DisplayMatches(im1, im2, matched_pairs):
    """Display matches on a new image with the two input images placed side by side.

    Arguments:
     im1           1st image (in PIL 'RGB' format)
     im2           2nd image (in PIL 'RGB' format)
     matched_pairs list of matching keypoints, im1 to im2

    Displays and returns a newly created image (in PIL 'RGB' format)
    """
    im3 = AppendImages(im1,im2)
    offset = im1.size[0]
    draw = ImageDraw.Draw(im3)
    for match in matched_pairs:
        draw.line((match[0][1], match[0][0], offset+match[1][1], match[1][0]),fill="red",width=2)
    im3.show()
    return im3

def match(image1,image2):
    """Input two images and their associated SIFT keypoints.
    Display lines connecting the first 5 keypoints from each image.
    Note: These 5 are not correct matches, just randomly chosen points.

    The arguments image1 and image2 are file names without file extensions.

    Returns the number of matches displayed.

    Example: match('scene','book')
    """
    im1, keypoints1, descriptors1 = ReadKeys('/Users/zhangquan/Documents/CS425/hw4/assign/'+image1)
    im2, keypoints2, descriptors2 = ReadKeys('/Users/zhangquan/Documents/CS425/hw4/assign/'+image2)
    #
    # REPLACE THIS CODE WITH YOUR SOLUTION (ASSIGNMENT 5, QUESTION 3)
    #
    #Generate five random matches (for testing purposes)
    matched_pairs = []
    threshold = 0.65
    
    for i in range(len(descriptors1)):
        angles = []
        for j in range(len(descriptors2)):
            # Compare each descriptor in descriptors1 with 
            # every descriptor in descriptors2, calculating the 
            # angle between each pair
            dot = np.dot(descriptors1[i], descriptors2[j])
            angles.append(math.acos(dot))

        # Sort the resulting angles 
        # and select the two smallest
        sortedAngles = sorted(angles)
        first = sortedAngles[0]
        second = sortedAngles[1]

        # We calculate the ratio between the best and second best angles, 
        # and only count it as a match if the ratio is less than our threshold
        if (first/second <= threshold):
            bestIndex = angles.index(first)
            matched_pairs.append([keypoints1[i], keypoints2[bestIndex]])

    # maximum angle difference in degrees between change in orientation. Used in RANSAC
    orientationLimit = 21
    # maxiumum scale ratio. Used in RANSAC
    scaleLimit = 0.9
    largestSupport = []
    
    for i in range(10):
        randomMatch = matched_pairs[np.random.randint(len(matched_pairs))]
        deltaScale1 = abs(randomMatch[0][2] - randomMatch[1][2])
        deltaOrientation1 = randomMatch[0][3] - randomMatch[1][3]
        currentSupport = []
        
        for match in matched_pairs:
            # check for consistency and add it to currentSupport if consistent
            deltaOrientation2 = match[0][3] - match[1][3]
            difference = absoluteAngleDifference(deltaOrientation1, deltaOrientation2)
            if (difference > orientationLimit):
                # Is not consistent, move on to next element
                continue

            # Check scale
            deltaScale2 = abs(match[0][2] - match[1][2])
            maxScale = max(deltaScale1, deltaScale2)
            minScale = min(deltaScale1, deltaScale2)
            if (maxScale * scaleLimit > minScale):
                continue
            # if we reach this point, it's consistent, add it to our support set
            currentSupport.append(match)

        # Update the largest support set
        if (len(currentSupport) > len(largestSupport)):
            largestSupport = currentSupport

    print ("length of largest support set: ")
    print (len(largestSupport))
    #
    # END OF SECTION OF CODE TO REPLACE
    #
    im3 = DisplayMatches(im1, im2, matched_pairs)
    return im3
    
def absoluteAngleDifference(a0, a1):
    '''
    Returns the absolute difference (in degrees) between two angles (specified
        in radians)
    a0 - an angle specified in radians
    a1 - an angle specified in radians
    '''
    anglea0 = 360+math.degrees(a0)
    anglea1 = 360+math.degrees(a1)
    result = abs(anglea0 - anglea1)%180
    return result

#Test run...
match('library','library2')
#match('scene','basmati')
#match('scene','box')
