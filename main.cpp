#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

inline float distance(const Point2f &l_p1, const Point2f &l_p2, const Point2f &p)
{
    if((l_p1.x - l_p2.x) < 1e-4)    // vertical line
    {
        return fabs(p.x - l_p1.x/2.0 - l_p2.x/2.0);
    }
    else
    {
        float k = (l_p2.y - l_p1.y)/(l_p2.x - l_p1.x);
        float b = l_p1.y - k*l_p1.x;
        return fabs(k*p.x - p.y + b)/sqrt(1 + k*k);
    }
}

inline bool all_zeros(const Vec3b &v)
{
    return v[0] == 0 && v[1] == 0 && v[2] == 0;
}
inline bool far_greater(const Vec3b &lhs, const Vec3b &rhs)
{
    return lhs[0] > 2*rhs[0] || lhs[1] > 2*rhs[1] || lhs[2] > 2*rhs[2];
}
inline bool far_lower(const Vec3b &lhs, const Vec3b &rhs)
{
    return rhs[0] > 2*lhs[0] || rhs[1] > 2*lhs[1] || rhs[2] > 2*lhs[2];
}
void get_points(vector<Point2f> &pAs, vector<Point2f> &pBs,
                vector<Point2f> &pCs, vector<Point2f> &pDs)
{
    ifstream file("ps.dat");
    if(!file.is_open())
    {
        cout << "file open failed" << endl;
        exit(-1);
    }
    pAs.clear();
    pBs.clear();
    pCs.clear();
    pDs.clear();
    Point2f pA, pB, pC, pD;
    while(file >> pA.x >> pA.y >> pB.x >> pB.y >> pC.x >> pC.y >> pD.x >> pD.y)
    {
        pAs.push_back(pA);
        pBs.push_back(pB);
        pCs.push_back(pC);
        pDs.push_back(pD);
    }
}

void merge_pics(vector<Mat> &pics, Mat &dst)
{
    pics[0].copyTo(dst);
    size_t w = dst.cols;
    size_t h = dst.rows;

    vector<Point2f> pAs, pBs, pCs, pDs;
    get_points(pAs, pBs, pCs, pDs);

    for(size_t i = 1; i < pics.size(); i++)
    {
        for(size_t x = 0; x < w; x++)
        {
            for(size_t y = 0; y < h; y++)
            {
                Point2f p(x, y);
                Vec3b dst_p = dst.at<Vec3b>(p);
                Vec3b pic_p = pics[i].at<Vec3b>(p);
                if (!all_zeros(dst_p) && !all_zeros(pic_p))
                {
                    if(far_greater(dst_p, pic_p))
                    {
                        dst.at<Vec3b>(p) = dst_p;
                    }
                    else if(far_lower(dst_p, pic_p))
                    {
                        dst.at<Vec3b>(p) = pic_p;
                    }
                    else
                    {
                        float d1 = distance(pAs[i], pDs[i], p);
                        float d2 = distance(pBs[i-1], pCs[i-1], p);
                        float r = d1/(d1 + d2);
                        dst.at<Vec3b>(p) = dst_p*(1 - r) + r*pic_p;
                    }
                }
                else if(all_zeros(dst.at<Vec3b>(p)) && !all_zeros(pics[i].at<Vec3b>(p)))
                {
                    dst.at<Vec3b>(p) = pics[i].at<Vec3b>(p);
                }
            }
        }
    }
}

int main()
{
    const size_t num = 4;
    vector<Mat> pics(num);
    for(size_t i = 0; i < num; i++)
    {
        pics[i] = imread("result_" + to_string(i) + ".jpg");
        if(pics[i].empty())
        {
            cout << "read files failed" << endl;
            return -1;
        }
    }
    Mat dst;
    merge_pics(pics, dst);
    namedWindow("result", CV_WINDOW_NORMAL);
    imshow("result", dst);
    imwrite("result.jpg", dst);
    waitKey();
    return 0;
}
