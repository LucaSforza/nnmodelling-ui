#include "OrthogonalRouter.hpp"

#include <QLineF>
#include <QtTest>

namespace {
using namespace OrthogonalRouter;

Request basicRequest() {
    Request request;
    request.edgeId = QStringLiteral("edge");
    request.source = {{50, 60}, {0, 1}, QRectF(0, 0, 100, 60), QStringLiteral("source")};
    request.target = {{50, 220}, {0, -1}, QRectF(0, 220, 100, 60), QStringLiteral("target")};
    request.obstacles = {{request.source.ownerBounds, request.source.ownerId},
                         {request.target.ownerBounds, request.target.ownerId}};
    return request;
}

bool orthogonal(const QVector<QPointF> &points) {
    for (int i = 1; i < points.size(); ++i)
        if (qAbs(points[i].x() - points[i - 1].x()) > 1e-6 &&
            qAbs(points[i].y() - points[i - 1].y()) > 1e-6) return false;
    return true;
}

bool interiorCrossed(const QPointF &a, const QPointF &b, const QRectF &rect) {
    if (qAbs(a.x() - b.x()) < 1e-6) {
        if (a.x() <= rect.left() || a.x() >= rect.right()) return false;
        return qMax(qMin(a.y(), b.y()), rect.top()) < qMin(qMax(a.y(), b.y()), rect.bottom());
    }
    if (a.y() <= rect.top() || a.y() >= rect.bottom()) return false;
    return qMax(qMin(a.x(), b.x()), rect.left()) < qMin(qMax(a.x(), b.x()), rect.right());
}

bool clears(const QVector<QPointF> &points, const QRectF &obstacle, int firstSegment = 1) {
    for (int i = firstSegment; i + 1 < points.size(); ++i)
        if (interiorCrossed(points[i], points[i + 1], obstacle)) return false;
    return true;
}

bool usesNearbyParallelTrack(const QVector<QPointF> &first, const QVector<QPointF> &second) {
    for (int i = 1; i < first.size(); ++i) {
        const QPointF a = first[i - 1], b = first[i];
        for (int j = 1; j < second.size(); ++j) {
            const QPointF c = second[j - 1], d = second[j];
            if (qAbs(a.x() - b.x()) < 1e-6 && qAbs(c.x() - d.x()) < 1e-6 &&
                qAbs(a.x() - c.x()) <= 7.0) {
                const qreal overlap = qMin(qMax(a.y(), b.y()), qMax(c.y(), d.y())) -
                                      qMax(qMin(a.y(), b.y()), qMin(c.y(), d.y()));
                if (overlap > 20.0) return true;
            }
            if (qAbs(a.y() - b.y()) < 1e-6 && qAbs(c.y() - d.y()) < 1e-6 &&
                qAbs(a.y() - c.y()) <= 7.0) {
                const qreal overlap = qMin(qMax(a.x(), b.x()), qMax(c.x(), d.x())) -
                                      qMax(qMin(a.x(), b.x()), qMin(c.x(), d.x()));
                if (overlap > 20.0) return true;
            }
        }
    }
    return false;
}
}

class QtRoutingTest final : public QObject {
    Q_OBJECT
private slots:
    void producesOrthogonalPolylineAroundObstacles() {
        Request request = basicRequest();
        const QRectF obstacle(35, 110, 30, 50);
        request.obstacles.append({obstacle, QStringLiteral("blocker")});
        const auto points = route(request);
        QVERIFY(points.size() >= 4);
        QCOMPARE(points.first(), request.source.point);
        QCOMPARE(points.last(), request.target.point);
        QVERIFY(orthogonal(points));
        QVERIFY(clears(points, obstacle.adjusted(-request.margin, -request.margin,
                                                  request.margin, request.margin)));
    }

    void routesMustLeaveAndEnterAlongPortDirection() {
        Request request = basicRequest();
        request.source.outward = {1, 0};
        request.source.point = {100, 30};
        request.target.outward = {-1, 0};
        request.target.ownerBounds = QRectF(400, 220, 100, 60);
        request.target.point = {400, 250};
        request.obstacles[1].bounds = request.target.ownerBounds;
        const auto points = route(request);
        QVERIFY(points.size() >= 2);
        QCOMPARE(points[1].y(), points.first().y());
        QVERIFY(points[1].x() > points.first().x());
        QCOMPARE(points[points.size() - 2].y(), points.last().y());
        QVERIFY(points[points.size() - 2].x() < points.last().x());
        QVERIFY(orthogonal(points));
    }

    void obstacleChangeReroutesExistingEdge() {
        Request request = basicRequest();
        request.source.point = {25, 60};
        request.target.point = {25, 220};
        const auto original = route(request);
        QVERIFY(original.size() >= 3);
        const QPointF middle((original[1].x() + original[2].x()) / 2,
                             (original[1].y() + original[2].y()) / 2);
        QRectF blocker(middle.x() - 12, middle.y() - 12, 24, 24);
        request.obstacles.append({blocker, QStringLiteral("moved-node")});
        const auto rerouted = route(request);
        QVERIFY(rerouted.size() >= 2);
        QVERIFY(orthogonal(rerouted));
        QVERIFY(clears(rerouted, blocker.adjusted(-request.margin, -request.margin,
                                                   request.margin, request.margin)));
        QVERIFY(rerouted != original);
    }

    void existingRoutesPreferParallelTracks() {
        Request request = basicRequest();
        const auto first = route(request);
        QVERIFY(first.size() >= 4);
        request.edgeId = QStringLiteral("second-edge");
        request.existingRoutes.append(first);
        const auto second = route(request);
        QVERIFY(second.size() >= 4);
        QVERIFY(orthogonal(second));
        QVERIFY(second != first);
        QVERIFY(usesNearbyParallelTrack(first, second));
        QVERIFY(second[1] == first[1]); // The port escape remains a shared stem.
    }

    void blockedPortEscapeFailsWithoutUnsafeFallback() {
        Request request = basicRequest();
        request.obstacles.append({QRectF(45, 65, 10, 20), QStringLiteral("escape-blocker")});
        QVERIFY(route(request).isEmpty());
    }
};

QTEST_MAIN(QtRoutingTest)
#include "qt_routing_test.moc"
