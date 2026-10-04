#ifndef NN_ORTHOGONAL_ROUTER_HPP
#define NN_ORTHOGONAL_ROUTER_HPP

#include <QPointF>
#include <QRectF>
#include <QString>
#include <QVector>

namespace OrthogonalRouter {

struct Endpoint {
    QPointF point;
    QPointF outward;
    QRectF ownerBounds;
    QString ownerId;
};

struct Obstacle {
    QRectF bounds;
    QString id;
};

struct Request {
    QString edgeId;
    Endpoint source;
    Endpoint target;
    QVector<Obstacle> obstacles;
    QVector<QVector<QPointF>> existingRoutes;
    qreal margin = 10.0;
};

// Returns an orthogonal polyline, or an empty vector when no safe path exists.
QVector<QPointF> route(const Request &request);

}

#endif
