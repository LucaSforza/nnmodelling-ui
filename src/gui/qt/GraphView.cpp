#include "GraphView.hpp"

#include "GraphScene.hpp"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QScrollBar>
#include <QTransform>
#include <QWheelEvent>

GraphView::GraphView(GraphScene *scene, QWidget *parent) : QGraphicsView(scene, parent) {
    setRenderHint(QPainter::Antialiasing);
    setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
    setTransformationAnchor(QGraphicsView::NoAnchor);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setDragMode(QGraphicsView::RubberBandDrag);
    setRubberBandSelectionMode(Qt::IntersectsItemShape);
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    setFrameShape(QFrame::NoFrame);
    setFocusPolicy(Qt::StrongFocus);
}

void GraphView::fitGraph() {
    if (!scene() || scene()->items().isEmpty()) return;
    auto *graph = qobject_cast<GraphScene *>(scene());
    if (graph) graph->recomputeRoutes();
    const QRectF bounds = (graph ? graph->contentBounds() : scene()->itemsBoundingRect())
                              .adjusted(-50, -50, 50, 50);
    if (bounds.isEmpty() || viewport()->width() <= 0 || viewport()->height() <= 0) return;
    zoom_ = qMin<qreal>(3.0, qMin(viewport()->width() / bounds.width(),
                                  viewport()->height() / bounds.height()));
    if (!(zoom_ > 0.0)) return;
    minimumZoom_ = qMin<qreal>(0.2, zoom_);
    setTransform(QTransform::fromScale(zoom_, zoom_));
    const qreal horizontalReach = viewport()->width() / zoom_;
    const qreal verticalReach = viewport()->height() / zoom_;
    scene()->setSceneRect(bounds.adjusted(-horizontalReach, -verticalReach,
                                           horizontalReach, verticalReach));
    centerOn(bounds.center());
}

void GraphView::focusAtTop(const QRectF &bounds) {
    if (bounds.isEmpty() || viewport()->width() <= 0 || viewport()->height() <= 0) return;
    zoom_ = 0.7;
    minimumZoom_ = 0.2;
    setTransform(QTransform::fromScale(zoom_, zoom_));
    const QPointF center(bounds.center().x(),
                         bounds.center().y() + viewport()->height() / (4.0 * zoom_));
    const qreal horizontalReach = viewport()->width() / zoom_;
    const qreal verticalReach = viewport()->height() / zoom_;
    const QRectF cameraRect(center.x() - horizontalReach, center.y() - verticalReach,
                            horizontalReach * 2.0, verticalReach * 2.0);
    scene()->setSceneRect(cameraRect.united(
        scene()->itemsBoundingRect().adjusted(-1.0, -1.0, 1.0, 1.0)));
    centerOn(center);
}

void GraphView::zoomIn() { zoomAt(1.15, viewport()->rect().center()); }

void GraphView::zoomOut() { zoomAt(1.0 / 1.15, viewport()->rect().center()); }

void GraphView::zoomAt(qreal requestedFactor, const QPoint &viewportPosition) {
    const qreal next = qBound<qreal>(minimumZoom_, zoom_ * requestedFactor, 3.0);
    if (qFuzzyCompare(next, zoom_)) return;
    const QPointF scenePositionBefore = mapToScene(viewportPosition);
    const qreal factor = next / zoom_;
    zoom_ = next;
    scale(factor, factor);
    const QPoint mappedAnchor = mapFromScene(scenePositionBefore);
    horizontalScrollBar()->setValue(horizontalScrollBar()->value() +
                                    mappedAnchor.x() - viewportPosition.x());
    verticalScrollBar()->setValue(verticalScrollBar()->value() +
                                  mappedAnchor.y() - viewportPosition.y());
}

void GraphView::wheelEvent(QWheelEvent *event) {
    const qreal requestedFactor = event->angleDelta().y() > 0 ? 1.15 : 1.0 / 1.15;
    const QPoint viewportPosition = event->position().toPoint();
    zoomAt(requestedFactor, viewportPosition);
    event->accept();
}

void GraphView::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::MiddleButton || (spaceDown_ && event->button() == Qt::LeftButton)) {
        panning_ = true;
        lastPan_ = event->pos();
        viewport()->setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);
}

void GraphView::mouseMoveEvent(QMouseEvent *event) {
    if (panning_) {
        const QPoint delta = event->pos() - lastPan_;
        lastPan_ = event->pos();
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
        verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
        event->accept();
        return;
    }
    QGraphicsView::mouseMoveEvent(event);
}

void GraphView::mouseReleaseEvent(QMouseEvent *event) {
    if (panning_ && (event->button() == Qt::MiddleButton || event->button() == Qt::LeftButton)) {
        panning_ = false;
        viewport()->unsetCursor();
        event->accept();
        return;
    }
    QGraphicsView::mouseReleaseEvent(event);
}

void GraphView::keyPressEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Escape) {
        if (auto *graph = qobject_cast<GraphScene *>(scene())) graph->cancelInteraction();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Space && !event->isAutoRepeat()) {
        spaceDown_ = true;
        event->accept();
        return;
    }
    QGraphicsView::keyPressEvent(event);
}

void GraphView::keyReleaseEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Space && !event->isAutoRepeat()) {
        spaceDown_ = false;
        event->accept();
        return;
    }
    QGraphicsView::keyReleaseEvent(event);
}
