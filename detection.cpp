#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <queue>
#include <Eigen/Dense>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>

// ───────────────── CONFIG ─────────────────
const float PLANE_DIST_THR = 0.02f;

const float SEED_THR       = -0.025f;
const float EXPAND_THR     = -0.01f;
const float LOCAL_DIFF_THR = 0.015f;

const int   K              = 20;

const float DBSCAN_EPS     = 0.13f;
const int   DBSCAN_MIN     = 50;

const float MAX_SLOPE_COS  = 0.94f;
// ──────────────────────────────────────────


// ───────── DBSCAN ─────────
std::vector<int> dbscan(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree,
    float eps, int minPts)
{
    int n = cloud->size();
    std::vector<int> labels(n, -1);
    int cid = 0;

    for (int i = 0; i < n; i++) {
        if (labels[i] != -1) continue;

        std::vector<int> nb;
        std::vector<float> dist;
        tree->radiusSearch((*cloud)[i], eps, nb, dist);

        if ((int)nb.size() < minPts) {
            labels[i] = -2;
            continue;
        }

        std::queue<int> q;
        for (int x : nb) q.push(x);
        labels[i] = cid;

        while (!q.empty()) {
            int cur = q.front(); q.pop();

            if (labels[cur] == -2)
                labels[cur] = cid;

            if (labels[cur] != -1)
                continue;

            labels[cur] = cid;

            std::vector<int> nb2;
            std::vector<float> d2;
            tree->radiusSearch((*cloud)[cur], eps, nb2, d2);

            if ((int)nb2.size() >= minPts)
                for (int x : nb2) q.push(x);
        }

        cid++;
    }

    return labels;
}


// ───────── MAIN ─────────
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./detect input.ply\n";
        return -1;
    }

    // ── Load cloud ──
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(argv[1], *cloud);
    int N = cloud->size();

    std::cout << "Number of points: " << N << "\n";

    // ── Plane fitting ──
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(PLANE_DIST_THR);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coeff);

    float a = coeff->values[0];
    float b = coeff->values[1];
    float c = coeff->values[2];
    float d = coeff->values[3];

    float norm = std::sqrt(a*a + b*b + c*c);

    Eigen::Vector3f plane_n(a,b,c);
    plane_n.normalize();

    // ── KNN ──
    pcl::search::KdTree<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<std::vector<int>> knn(N);
    for (int i = 0; i < N; i++) {
        std::vector<int> idx(K);
        std::vector<float> d2(K);
        kdtree.nearestKSearch((*cloud)[i], K, idx, d2);
        knn[i] = idx;
    }

    // ── Distance + slope filter ──
    std::vector<float> dist(N);
    std::vector<bool> valid(N, true);

    for (int i = 0; i < N; i++) {
        auto& p = (*cloud)[i];
        dist[i] = (a*p.x + b*p.y + c*p.z + d) / norm;
    }

    // ── Local smoothness ──
    std::vector<bool> local(N,false);
    for (int i = 0; i < N; i++) {
        float mean = 0;
        for (int nb : knn[i]) mean += dist[nb];
        mean /= K;

        if (dist[i] < mean - LOCAL_DIFF_THR)
            local[i] = true;
    }

    // ── Seed (FIXED) ──
    std::vector<bool> seed(N,false), expand(N,false);

    for (int i = 0; i < N; i++) {
        if (dist[i] < SEED_THR)
            seed[i] = true;
    }

    // ── Expansion (local used here, NOT in seed) ──
    for (int i = 0; i < N; i++) {
        if (seed[i]) continue;

        int cnt = 0;
        for (int nb : knn[i])
            if (seed[nb]) cnt++;

        if (cnt > 0.3*K && dist[i] < EXPAND_THR && local[i])
            expand[i] = true;
    }

    // ── Candidate cloud ──
    pcl::PointCloud<pcl::PointXYZ>::Ptr cand(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> map_idx;

    for (int i = 0; i < N; i++) {
        if (seed[i] || expand[i]) {
            cand->push_back((*cloud)[i]);
            map_idx.push_back(i);
        }
    }

    std::cout << "Candidate points: " << cand->size() << "\n";

    // ── DBSCAN ──
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cand);

    auto labels = dbscan(cand, tree, DBSCAN_EPS, DBSCAN_MIN);
    int max_label = *std::max_element(labels.begin(), labels.end());

    std::vector<std::array<uint8_t,3>> colors(N, {150,150,150});

    // ── Output ──
    for (int cid = 0; cid <= max_label; cid++) {

        std::vector<int> pts;
        for (int i = 0; i < labels.size(); i++)
            if (labels[i] == cid)
                pts.push_back(i);

        if (pts.size() < 40) continue;

        Eigen::Vector3f center(0,0,0);
        float max_depth = 0;
        float volume = 0;

        for (int i : pts) {
            int orig = map_idx[i];
            auto& p = (*cloud)[orig];

            center += Eigen::Vector3f(p.x,p.y,p.z);

            float depth = std::abs(dist[orig]);
            max_depth = std::max(max_depth, depth);

            float area = DBSCAN_EPS * DBSCAN_EPS / pts.size();
            volume += depth * area;

            colors[orig] = {255,0,0};
        }

        center /= pts.size();
        float liters = volume * 1000.0f;

        printf(
            "Pothole #%d | Points: %zu | Center: (%.3f, %.3f, %.3f) | "
            "MaxDepth: %.4f m | Volume: %.4f m^3 | %.2f L\n",
            cid, pts.size(),
            center.x(), center.y(), center.z(),
            max_depth, volume, liters
        );
    }

    // ── Save ──
    std::ofstream out("detected.ply");

    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << N << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    out << "end_header\n";

    for (int i = 0; i < N; i++) {
        auto& p = (*cloud)[i];
        out << p.x << " " << p.y << " " << p.z << " "
            << (int)colors[i][0] << " "
            << (int)colors[i][1] << " "
            << (int)colors[i][2] << "\n";
    }

    std::cout << "Saved detected.ply\n";
}
