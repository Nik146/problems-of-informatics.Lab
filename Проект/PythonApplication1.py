import numpy
import cv2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = numpy.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        numpy.savetxt(f, verts, '%f %f %f %d %d %d')



def update(val = 0): 

    blockSize = cv2.getTrackbarPos('window_size', 'disparity')
    uniquenessRatio  = cv2.getTrackbarPos('uniquenessRatio', 'disparity')
    speckleWindowSize  = cv2.getTrackbarPos('speckleWindowSize', 'disparity')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disparity')
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disparity')
    min_disp = cv2.getTrackbarPos('min_disp', 'disparity')
    #num_disp = cv2.getTrackbarPos('num_disp', 'disparity')
  

    stereo = cv2.StereoSGBM(
        minDisparity = min_disp,
        numDisparities = num_disp,
        SADWindowSize = window_size,
        uniquenessRatio = uniquenessRatio,
        speckleRange = speckleRange,
        speckleWindowSize = speckleWindowSize,
        disp12MaxDiff = disp12MaxDiff,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        fullDP = False
    )
    
    print 'UPDATE!'
    disp = stereo.compute(imgL, imgR).astype(numpy.float32) / 16.0

    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)

    cv2.waitKey()

    print 'Creating point cloud...',
    h, w = imgL.shape[:2]
    f = 0.5*w                          
    Q = numpy.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], 
                    [0, 0, 0,     -f], 
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'yes.ply'
    write_ply('yes.ply', out_points, out_colors)
    print '%s saved' % 'yes.ply'


  

if __name__ == '__main__':

    print 'PRESS ENTER to CREATE point cloud'
    imgL = cv2.pyrDown( cv2.imread('/Python27/123/1.jpg') ) 
    imgR = cv2.pyrDown( cv2.imread('/Python27/123/2.jpg') )

    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 3
    speckleWindowSize = 3
    disp12MaxDiff = 200

    cv2.namedWindow('disparity')
    cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)    
    cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
    cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200, update)
    cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, update)
    cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)
    cv2.createTrackbar('min_disp', 'disparity', min_disp , 100, update)
    #cv2.createTrackbar('num_disp', 'disparity', num_disp , 500, update)

    update()
    cv2.waitKey()
    cv2.destroyAllWindows()
