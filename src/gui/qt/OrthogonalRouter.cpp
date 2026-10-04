#include "OrthogonalRouter.hpp"

#include <QLineF>
#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

namespace OrthogonalRouter {
namespace {
constexpr qreal kClearance = 1.0;
constexpr qreal kTrackOffset = 6.0;
constexpr qreal kEpsilon = 1e-7;
enum Direction { None = 0, Horizontal = 1, Vertical = 2 };

struct Cost {
    qreal overlap = 0;
    int crossings = 0;
    int bends = 0;
    qreal length = 0;
};

bool lessCost(const Cost &a, const Cost &b) {
    if (a.overlap != b.overlap) return a.overlap < b.overlap;
    if (a.crossings != b.crossings) return a.crossings < b.crossings;
    if (a.bends != b.bends) return a.bends < b.bends;
    return a.length < b.length;
}

bool equalCost(const Cost &a, const Cost &b) {
    return a.overlap == b.overlap && a.crossings == b.crossings &&
           a.bends == b.bends && a.length == b.length;
}

QPointF unit(const QPointF &direction) {
    if (qAbs(direction.x()) > 0.5 && qAbs(direction.y()) < 0.5)
        return QPointF(direction.x() > 0 ? 1 : -1, 0);
    if (qAbs(direction.y()) > 0.5 && qAbs(direction.x()) < 0.5)
        return QPointF(0, direction.y() > 0 ? 1 : -1);
    return {};
}

qreal escapeDistance(const QPointF &point, const QPointF &direction,
                     const QRectF &bounds, qreal margin) {
    const QRectF expanded = bounds.normalized().adjusted(-margin - kClearance,
        -margin - kClearance, margin + kClearance, margin + kClearance);
    if (direction.x() > 0) return qMax<qreal>(0, expanded.right() - point.x());
    if (direction.x() < 0) return qMax<qreal>(0, point.x() - expanded.left());
    if (direction.y() > 0) return qMax<qreal>(0, expanded.bottom() - point.y());
    if (direction.y() < 0) return qMax<qreal>(0, point.y() - expanded.top());
    return 0;
}

bool blocked(const QPointF &a, const QPointF &b, const QVector<QRectF> &rectangles) {
    for (const QRectF &rect : rectangles) {
        if (qAbs(a.x() - b.x()) < kEpsilon) {
            const qreal x = a.x();
            if (x <= rect.left() + kEpsilon || x >= rect.right() - kEpsilon) continue;
            if (qMax(qMin(a.y(), b.y()), rect.top()) <
                qMin(qMax(a.y(), b.y()), rect.bottom()) - kEpsilon) return true;
        } else {
            const qreal y = a.y();
            if (y <= rect.top() + kEpsilon || y >= rect.bottom() - kEpsilon) continue;
            if (qMax(qMin(a.x(), b.x()), rect.left()) <
                qMin(qMax(a.x(), b.x()), rect.right()) - kEpsilon) return true;
        }
    }
    return false;
}

bool perpendicularCrossing(const QPointF &a, const QPointF &b,
                           const QPointF &c, const QPointF &d) {
    if (qAbs(a.y() - b.y()) < kEpsilon && qAbs(c.x() - d.x()) < kEpsilon) {
        return c.x() >= qMin(a.x(), b.x()) - kEpsilon && c.x() < qMax(a.x(), b.x()) - kEpsilon &&
               a.y() > qMin(c.y(), d.y()) + kEpsilon && a.y() < qMax(c.y(), d.y()) - kEpsilon;
    }
    if (qAbs(a.x() - b.x()) < kEpsilon && qAbs(c.y() - d.y()) < kEpsilon) {
        return d.y() >= qMin(a.y(), b.y()) - kEpsilon && d.y() < qMax(a.y(), b.y()) - kEpsilon &&
               a.x() > qMin(c.x(), d.x()) + kEpsilon && a.x() < qMax(c.x(), d.x()) - kEpsilon;
    }
    return false;
}

Cost routeConflict(const QPointF &a, const QPointF &b,
                   const QVector<QVector<QPointF>> &routes) {
    Cost conflict;
    for (const QVector<QPointF> &route : routes) {
        for (int i = 1; i < route.size(); ++i) {
            const QPointF c = route[i - 1], d = route[i];
            if ((qAbs(a.x() - b.x()) < kEpsilon && qAbs(c.x() - d.x()) < kEpsilon &&
                 qAbs(a.x() - c.x()) < kEpsilon) ||
                (qAbs(a.y() - b.y()) < kEpsilon && qAbs(c.y() - d.y()) < kEpsilon &&
                 qAbs(a.y() - c.y()) < kEpsilon)) {
                const qreal overlap = qAbs(a.x() - b.x()) < kEpsilon
                    ? qMin(qMax(a.y(), b.y()), qMax(c.y(), d.y())) -
                      qMax(qMin(a.y(), b.y()), qMin(c.y(), d.y()))
                    : qMin(qMax(a.x(), b.x()), qMax(c.x(), d.x())) -
                      qMax(qMin(a.x(), b.x()), qMin(c.x(), d.x()));
                if (overlap > kEpsilon) conflict.overlap += overlap;
            } else if (perpendicularCrossing(a, b, c, d)) {
                ++conflict.crossings;
            }
        }
    }
    return conflict;
}

void appendUnique(QVector<qreal> *values, qreal value) {
    for (qreal present : *values) if (qAbs(present - value) < kEpsilon) return;
    values->append(value);
}

void simplify(QVector<QPointF> *points) {
    for (int i = points->size() - 2; i > 0; --i) {
        const QPointF before = points->at(i - 1), here = points->at(i), after = points->at(i + 1);
        if ((qAbs(before.x() - here.x()) < kEpsilon && qAbs(here.x() - after.x()) < kEpsilon) ||
            (qAbs(before.y() - here.y()) < kEpsilon && qAbs(here.y() - after.y()) < kEpsilon))
            points->removeAt(i);
    }
}

struct QueueEntry { Cost cost; int state; quint64 order; };
struct Later {
    bool operator()(const QueueEntry &a, const QueueEntry &b) const {
        if (!equalCost(a.cost, b.cost)) return lessCost(b.cost, a.cost);
        return a.order > b.order;
    }
};
}

QVector<QPointF> route(const Request &request) {
    QVector<QPointF> result;
    const QPointF sourceDirection = unit(request.source.outward);
    const QPointF targetDirection = unit(request.target.outward);
    if (sourceDirection.isNull() || targetDirection.isNull() ||
        !std::isfinite(request.margin) || request.margin < 0 ||
        !std::isfinite(request.source.point.x()) || !std::isfinite(request.source.point.y()) ||
        !std::isfinite(request.target.point.x()) || !std::isfinite(request.target.point.y()) ||
        request.source.ownerBounds.isEmpty() || request.target.ownerBounds.isEmpty()) return result;

    const qreal margin = request.margin;
    const QPointF start = request.source.point + sourceDirection *
        escapeDistance(request.source.point, sourceDirection, request.source.ownerBounds, margin);
    const QPointF finish = request.target.point + targetDirection *
        escapeDistance(request.target.point, targetDirection, request.target.ownerBounds, margin);
    QVector<QRectF> rectangles;
    rectangles.reserve(request.obstacles.size());
    for (const Obstacle &obstacle : request.obstacles) {
        if (!obstacle.bounds.isEmpty())
            rectangles.append(obstacle.bounds.normalized().adjusted(-margin, -margin, margin, margin));
    }
    bool hasSourceOwner = false, hasTargetOwner = false;
    for (const Obstacle &obstacle : request.obstacles) {
        hasSourceOwner |= obstacle.id == request.source.ownerId;
        hasTargetOwner |= obstacle.id == request.target.ownerId;
    }
    if (!hasSourceOwner)
        rectangles.append(request.source.ownerBounds.normalized().adjusted(-margin, -margin, margin, margin));
    if (!hasTargetOwner && request.target.ownerId != request.source.ownerId)
        rectangles.append(request.target.ownerBounds.normalized().adjusted(-margin, -margin, margin, margin));

    for (const Obstacle &obstacle : request.obstacles) {
        const QRectF expanded = obstacle.bounds.normalized().adjusted(-margin, -margin,
                                                                       margin, margin);
        if (obstacle.id != request.source.ownerId && blocked(request.source.point, start, {expanded}))
            return result;
        if (obstacle.id != request.target.ownerId && blocked(request.target.point, finish, {expanded}))
            return result;
    }

    QVector<qreal> xs, ys;
    appendUnique(&xs, start.x()); appendUnique(&xs, finish.x());
    appendUnique(&ys, start.y()); appendUnique(&ys, finish.y());
    for (const QRectF &rect : rectangles) {
        appendUnique(&xs, rect.left() - kClearance); appendUnique(&xs, rect.right() + kClearance);
        appendUnique(&ys, rect.top() - kClearance); appendUnique(&ys, rect.bottom() + kClearance);
    }
    const QRectF trackArea(start, finish);
    const QRectF expandedTrackArea = trackArea.normalized().adjusted(-300, -300, 300, 300);
    for (const QVector<QPointF> &existing : request.existingRoutes) {
        for (int i = 1; i < existing.size(); ++i) {
            const QPointF a = existing[i - 1], b = existing[i];
            if (!QRectF(a, b).normalized().adjusted(-kClearance, -kClearance,
                                                    kClearance, kClearance)
                     .intersects(expandedTrackArea)) continue;
            if (qAbs(a.x() - b.x()) < kEpsilon) {
                appendUnique(&xs, a.x() - kTrackOffset); appendUnique(&xs, a.x() + kTrackOffset);
                appendUnique(&ys, a.y()); appendUnique(&ys, b.y());
            } else {
                appendUnique(&ys, a.y() - kTrackOffset); appendUnique(&ys, a.y() + kTrackOffset);
                appendUnique(&xs, a.x()); appendUnique(&xs, b.x());
            }
        }
    }
    std::sort(xs.begin(), xs.end()); std::sort(ys.begin(), ys.end());
    const int columns = xs.size(), rows = ys.size();
    if (!columns || !rows || qint64(columns) * rows > 100000) return result;
    auto xIndex = [&xs](qreal x) {
        for (int i = 0; i < xs.size(); ++i) if (qAbs(xs[i] - x) < kEpsilon) return i;
        return -1;
    };
    auto yIndex = [&ys](qreal y) {
        for (int i = 0; i < ys.size(); ++i) if (qAbs(ys[i] - y) < kEpsilon) return i;
        return -1;
    };
    const int startX = xIndex(start.x()), startY = yIndex(start.y());
    const int finishX = xIndex(finish.x()), finishY = yIndex(finish.y());
    if (startX < 0 || startY < 0 || finishX < 0 || finishY < 0) return result;
    const int startNode = startY * columns + startX;
    const int finishNode = finishY * columns + finishX;

    const int stateCount = columns * rows * 3;
    QVector<Cost> distances(stateCount);
    QVector<bool> reached(stateCount, false);
    QVector<int> previous(stateCount, -1);
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, Later> queue;
    const Direction sourceAxis = sourceDirection.x() == 0 ? Vertical : Horizontal;
    const Direction targetAxis = targetDirection.x() == 0 ? Vertical : Horizontal;
    const int initialState = startNode * 3 + sourceAxis;
    reached[initialState] = true;
    quint64 order = 0;
    queue.push({{}, initialState, order++});
    int finalState = -1;
    Cost finalCost;
    while (!queue.empty()) {
        const QueueEntry current = queue.top(); queue.pop();
        if (!equalCost(current.cost, distances[current.state])) continue;
        if (finalState >= 0 && !lessCost(current.cost, finalCost)) break;
        const int node = current.state / 3;
        const Direction priorDirection = Direction(current.state % 3);
        if (node == finishNode) {
            // Endpoint escapes are painted too: count their bends, not only
            // turns inside the visibility grid.
            Cost candidate = current.cost;
            candidate.bends += priorDirection != targetAxis ? 1 : 0;
            if (finalState < 0 || lessCost(candidate, finalCost)) {
                finalState = current.state;
                finalCost = candidate;
            }
            continue;
        }
        const int row = node / columns, column = node % columns;
        const QPointF point(xs[column], ys[row]);
        const auto visit = [&](int nextRow, int nextColumn, Direction direction) {
            const QPointF next(xs[nextColumn], ys[nextRow]);
            if (blocked(point, next, rectangles)) return;
            const qreal length = QLineF(point, next).length();
            const Cost conflict = routeConflict(point, next, request.existingRoutes);
            const int nextState = (nextRow * columns + nextColumn) * 3 + direction;
            Cost candidate = current.cost;
            candidate.overlap += conflict.overlap;
            candidate.crossings += conflict.crossings;
            candidate.bends += priorDirection != None && priorDirection != direction ? 1 : 0;
            candidate.length += length;
            if (reached[nextState] && !lessCost(candidate, distances[nextState])) return;
            distances[nextState] = candidate;
            reached[nextState] = true;
            previous[nextState] = current.state;
            queue.push({candidate, nextState, order++});
        };
        if (column > 0) visit(row, column - 1, Horizontal);
        if (column + 1 < columns) visit(row, column + 1, Horizontal);
        if (row > 0) visit(row - 1, column, Vertical);
        if (row + 1 < rows) visit(row + 1, column, Vertical);
    }
    if (finalState < 0) return result;

    QVector<QPointF> reversed;
    for (int state = finalState; state >= 0; state = previous[state])
        reversed.append(QPointF(xs[(state / 3) % columns], ys[(state / 3) / columns]));
    std::reverse(reversed.begin(), reversed.end());
    simplify(&reversed);
    result.append(request.source.point);
    if (result.last() != start) result.append(start);
    for (const QPointF &point : reversed)
        if (result.last() != point) result.append(point);
    if (result.last() != finish) result.append(finish);
    if (result.last() != request.target.point) result.append(request.target.point);
    for (int i = 1; i < result.size(); ++i) {
        if (qAbs(result[i].x() - result[i - 1].x()) > kEpsilon &&
            qAbs(result[i].y() - result[i - 1].y()) > kEpsilon) return {};
    }
    return result;
}

}
