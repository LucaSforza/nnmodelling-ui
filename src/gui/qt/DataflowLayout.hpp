#ifndef NN_GUI_DATAFLOW_LAYOUT_HPP
#define NN_GUI_DATAFLOW_LAYOUT_HPP

#include "NodeItem.hpp"

#include <QHash>
#include <QRectF>
#include <QString>
#include <QVector>

namespace DataflowLayout {

struct Node {
    QString id;
    QRectF bounds;
    qreal crossAxisAnchor = 0.0;
    bool terminal = false;
};

struct Edge {
    QString source;
    QString target;
};

// Returns grid-aligned item origins without changing graph/model coordinates.
QHash<QString, QPointF> arrange(const QVector<Node> &nodes,
                                const QVector<Edge> &edges,
                                FlowDirection direction);

} // namespace DataflowLayout

#endif
