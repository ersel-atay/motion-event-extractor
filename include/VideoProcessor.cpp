//
// Created by ATAY on 3.08.2026.
//

#include "VideoProcessor.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

VideoProcessor::VideoProcessor(const std::string &Path) {
    videoPath = Path;
}

std::vector<MotionEvent> VideoProcessor::process(double minContourArea,
                                                 double thresholdVal,
                                                 double minGapSec) {
    std::vector<MotionEvent> events;
    cv::VideoCapture capture(videoPath);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video file " << videoPath << std::endl;
        return events;
    }

    cv::Mat currentFrame, prevFrame, grayFrame, blurredFrame, diffFrame, threshFrame;
    bool isMoving = false;
    MotionEvent activeEvent;
    double lastTimestamp = 0.0;

    while (true) {
        capture >> currentFrame;
        if (currentFrame.empty()) {
            break;
        }
        double currentTimestamp = capture.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        lastTimestamp = currentTimestamp;
        cv::cvtColor(currentFrame, grayFrame, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(grayFrame, blurredFrame, cv::Size(21, 21), 0);

        if (prevFrame.empty()) {
            prevFrame = blurredFrame.clone();
            continue;
        }
        cv::absdiff(prevFrame, blurredFrame, diffFrame);
        cv::threshold(diffFrame, threshFrame, thresholdVal, 255, cv::THRESH_BINARY);
        cv::dilate(threshFrame, threshFrame, cv::Mat(), cv::Point(-1, -1), 2);

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(threshFrame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        bool hasMotionInThisFrame = false;
        cv::Rect currentBox;
        for (auto &contour: contours) {
            if (cv::contourArea(contour) > minContourArea) {
                hasMotionInThisFrame = true;
                currentBox = cv::boundingRect(contour);
                break;
            }
        }

        if (hasMotionInThisFrame) {
            if (!isMoving) {
                isMoving = true;
                activeEvent.startTimeSec = currentTimestamp;
                activeEvent.boundingBox = currentBox;
            }
        } else {
            if (isMoving) {
                activeEvent.endTimeSec = currentTimestamp;
                events.push_back(activeEvent);
                isMoving = false;
            }
        }
        prevFrame = blurredFrame.clone();
    }
    if (isMoving) {
        activeEvent.endTimeSec = lastTimestamp;
        events.push_back(activeEvent);
    }
    capture.release();
    return mergedEvents(events, minGapSec);
}

std::vector<MotionEvent> VideoProcessor::mergedEvents(std::vector<MotionEvent> events, double maxGapSec) {
    std::vector<MotionEvent> mergedEvents;
    for (auto &event: events) {
        if (mergedEvents.empty()) {
            mergedEvents.push_back(event);
        } else {
            auto lastEvent = mergedEvents.back();
            if ((event.startTimeSec - lastEvent.startTimeSec) <= maxGapSec) {
                lastEvent.startTimeSec = event.startTimeSec;
                lastEvent.boundingBox |= event.boundingBox;
            } else {
                mergedEvents.push_back(event);
            }
        }
    }
    return mergedEvents;
}
