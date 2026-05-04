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

// ───────────────── CONFIG ─────────────────
const float PLANE_DIST_THR = 0.02f;   // RANSAC inlier tolerance

const float SEED_THR       = -0.025f; // 2.5cm below plane = seed
const float EXPAND_THR     = -0.01f;  // 1cm below plane = expandable
const float LOCAL_DIFF_THR = 0.015f;  // local neighborhood difference

const int   K              = 20;      // KNN neighbors

const float DBSCAN_EPS     = 0.13f;   // 13cm cluster radius
const int   DBSCAN_MIN     = 50;      // min points per cluster
const int   MIN_CLUSTER_PTS= 40;      // min points to report

const float MAX_SLOPE_COS  = 0.94f;   // ~20 deg — rejects steep surfaces
// ──────────────────────────────────────────


// ───────── DBSCAN (KD-tree radius search) ─────────
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

        std::vector<int>   nb;
        std::vector<float> dist;
        tree->radiusSearch((*cloud)[i], eps, nb, dist);

        if ((int)nb.size() < minPts) {
            labels[i] = -2;  // noise
            continue;
        }

        std::queue<int> q;
        for (int x : nb) q.push(x);
        labels[i] = cid;

        while (!q.empty()) {
            int cur = q.front(); q.pop();

            if (labels[cur] == -2) labels[cur] = cid;
            if (labels[cur] != -1) continue;

            labels[cur] = cid;

            std::vector<int>   nb2;
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

    // ── 1. Load ──
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(argv[1], *cloud);
    int N = cloud->size();
    std::cout << "Loaded " << N << " points\n";

    // ── 2. RANSAC plane fit ──
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setOptimizeCoefficients(true);
    seg.setDistanceThreshold(PLANE_DIST_THR);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coeff);

    if (inliers->indices.empty()) {
        std::cerr << "No plane found!\n";
        return -1;
    }

    float a = coeff->values[0], b = coeff->values[1],
          c = coeff->values[2], d = coeff->values[3];
    float norm = std::sqrt(a*a + b*b + c*c);

    Eigen::Vector3f plane_n(a, b, c);
    plane_n.normalize();

    std::cout << "Plane: " << a << "x + " << b
              << "y + " << c << "z + " << d << " = 0\n";

    // ── 3. KNN ──
    pcl::search::KdTree<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<std::vector<int>> knn(N);
    for (int i = 0; i < N; i++) {
        std::vector<int>   idx(K);
        std::vector<float> d2(K);
        kdtree.nearestKSearch((*cloud)[i], K, idx, d2);
        knn[i] = idx;
    }

    // ── 4. Signed distances ──
    std::vector<float> dist(N);
    for (int i = 0; i < N; i++) {
        auto& p = (*cloud)[i];
        dist[i] = (a*p.x + b*p.y + c*p.z + d) / norm;
    }

    // ── 5. Slope filter — reject steep surfaces (curbs, walls) ──
    // Estimate local normal per point using KNN, compare to plane normal
    std::vector<bool> flat(N, true);
    for (int i = 0; i < N; i++) {
        // Build local covariance matrix from neighbors
        Eigen::Vector3f mean(0,0,0);
        for (int nb : knn[i]) {
            auto& p = (*cloud)[nb];
            mean += Eigen::Vector3f(p.x, p.y, p.z);
        }
        mean /= K;

        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (int nb : knn[i]) {
            auto& p = (*cloud)[nb];
            Eigen::Vector3f v(p.x - mean.x(), p.y - mean.y(), p.z - mean.z());
            cov += v * v.transpose();
        }

        // Smallest eigenvector = local normal
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        Eigen::Vector3f local_n = solver.eigenvectors().col(0);
        local_n.normalize();

        // If local normal deviates too much from road plane → steep surface
        float cos_angle = std::abs(local_n.dot(plane_n));
        if (cos_angle < MAX_SLOPE_COS)
            flat[i] = false;  // reject: wall, curb, steep edge
    }

    // ── 6. Local smoothness ──
    std::vector<bool> local(N, false);
    for (int i = 0; i < N; i++) {
        if (!flat[i]) continue;  // skip steep points
        float mean_d = 0;
        for (int nb : knn[i]) mean_d += dist[nb];
        mean_d /= K;
        if (dist[i] < mean_d - LOCAL_DIFF_THR)
            local[i] = true;
    }

    // ── 7. Seed mask ──
    std::vector<bool> seed(N, false);
    for (int i = 0; i < N; i++)
        if (flat[i] && dist[i] < SEED_THR)
            seed[i] = true;

    // ── 8. Expand mask ──
    std::vector<bool> expand(N, false);
    for (int i = 0; i < N; i++) {
        if (seed[i] || !flat[i]) continue;
        int cnt = 0;
        for (int nb : knn[i])
            if (seed[nb]) cnt++;
        if (cnt > 0.3*K && dist[i] < EXPAND_THR && local[i])
            expand[i] = true;
    }

    // ── 9. Candidate cloud ──
    pcl::PointCloud<pcl::PointXYZ>::Ptr cand(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> map_idx;

    for (int i = 0; i < N; i++) {
        if (seed[i] || expand[i]) {
            cand->push_back((*cloud)[i]);
            map_idx.push_back(i);
        }
    }
    std::cout << "Candidate points: " << cand->size() << "\n";

    if (cand->empty()) {
        std::cout << "No potholes detected\n";
        return 0;
    }

    // ── 10. DBSCAN ──
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cand);

    auto labels   = dbscan(cand, tree, DBSCAN_EPS, DBSCAN_MIN);
    int max_label = *std::max_element(labels.begin(), labels.end());

    std::vector<std::array<uint8_t,3>> colors(N, {150, 150, 150});
    int valid = 0;

    std::cout << "\n=== POTHOLE DETECTION RESULTS ===\n";

    for (int cid = 0; cid <= max_label; cid++) {
        std::vector<int> pts;
        for (int i = 0; i < (int)labels.size(); i++)
            if (labels[i] == cid)
                pts.push_back(i);

        if ((int)pts.size() < MIN_CLUSTER_PTS) continue;

        valid++;

        Eigen::Vector3f center(0,0,0);
        Eigen::Vector3f mn(1e9,1e9,1e9), mx(-1e9,-1e9,-1e9);
        float max_depth = 0, volume = 0;

        for (int i : pts) {
            int orig = map_idx[i];
            auto& p  = (*cloud)[orig];
            Eigen::Vector3f pv(p.x, p.y, p.z);

            center   += pv;
            mn        = mn.cwiseMin(pv);
            mx        = mx.cwiseMax(pv);

            float depth = std::abs(dist[orig]);
            max_depth   = std::max(max_depth, depth);

            float area = DBSCAN_EPS * DBSCAN_EPS / pts.size();
            volume    += depth * area;

            colors[orig] = {255, 0, 0};
        }

        center /= pts.size();
        float liters    = volume * 1000.0f;
        float depth_cm  = max_depth * 100.0f;
        float width_cm  = (mx.x() - mn.x()) * 100.0f;
        float length_cm = (mx.y() - mn.y()) * 100.0f;

        printf(
            "Pothole #%02d | Points: %zu | "
            "Center: (%.3f, %.3f, %.3f) | "
            "Depth: %.1f cm | Width: %.1f cm | Length: %.1f cm | "
            "Volume: %.4f m^3 (%.2f L)\n",
            valid, pts.size(),
            center.x(), center.y(), center.z(),
            depth_cm, width_cm, length_cm,
            volume, liters
        );
    }

    std::cout << "\nTotal potholes: " << valid << "\n";

    // ── 11. Save PLY ──
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

    std::cout << "Saved: detected.ply\n";
    return 0;
}