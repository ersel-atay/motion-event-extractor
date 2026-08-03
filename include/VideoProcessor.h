//
// Created by ATAY on 3.08.2026.
//

#ifndef MOTION_EVENT_EXTRACTOR_VIDEOPROCESSOR_H
#define MOTION_EVENT_EXTRACTOR_VIDEOPROCESSOR_H
#include <string>
#include <opencv2/videoio.hpp>

struct MotionEvent {
    double startTimeSec;
    double endTimeSec;
    int indexFrame;
    cv::Rect boundingBox;
};

class VideoProcessor {
public:
    VideoProcessor(const std::string &videoPath);

    std::vector<MotionEvent> process(double minContourArea = 500.0,
                                     double thresholdVal=25.0,
                                     double minGapSec=1.0);

private:
    std::string videoPath;
    cv::VideoCapture capture;

    std::vector<MotionEvent> mergedEvents(std::vector<MotionEvent> events, double maxGapSec = 1.0);
};
#endif //MOTION_EVENT_EXTRACTOR_VIDEOPROCESSOR_H
