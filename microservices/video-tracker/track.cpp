// Standard include files
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <cpp_redis/cpp_redis>
#include <iostream>
#include "json.hpp"

using namespace cv;
using namespace std;
using json = nlohmann::json;

int main(int argc, char **argv)
{

    string VIDEO_IP;
    string REDIS_HOST;
    int REDIS_PORT;

    VIDEO_IP = "http://10.35.114.8:8080/video";
    REDIS_HOST = "eil-computenode1.stanford.edu";
    REDIS_PORT = 8080;

    // Set up tracker.
    // Instead of MIL, you can also use
    // BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
    Ptr<Tracker> tracker = TrackerMIL::create();

    // Read video
    VideoCapture video(VIDEO_IP);
    //VideoCapture video("videos/chaplin.mp4");

    video.set(CV_CAP_PROP_BUFFERSIZE, 10);

    // Check video is open
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }

    // Read first frame.
    Mat frame;
    video.read(frame);

    // Uncomment the line below if you
    // want to choose the bounding box
    Rect2d bbox = selectROI(frame, false);

    // Initialize tracker with first frame and bounding box
    tracker->init(frame, bbox);

    // Redis Client
    cpp_redis::client client;

    client.connect(REDIS_HOST, REDIS_PORT);


    // Initialize the frame counter
    int count;
    json box;

    while(video.read(frame))
    {
        // Update tracking results
        tracker->update(frame, bbox);

        // Draw bounding box


        box["x"] = bbox.x;
        box["y"] = bbox.y;
        box["width"] = bbox.width;
        box["height"] = bbox.height;

        // Save to redis
        client.set("position", box.dump());
        client.commit();

        count++;
        cout << count << endl;

        // Display result
        if (count % 5 == 0){
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
            imshow("Tracking", frame);
            int k = waitKey(1);
            if(k == 27) break;
        }
    }

    return 0;
}