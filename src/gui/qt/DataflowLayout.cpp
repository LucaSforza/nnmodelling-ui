#include "DataflowLayout.hpp"

#include "model/model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace DataflowLayout {
namespace {

constexpr qreal LaneGap = 80.0;
constexpr qreal RankGap = 60.0;
constexpr qreal Margin = 80.0;

qreal snap(qreal value) {
    return std::round(value / NN_MODEL_GRID_SPACING) * NN_MODEL_GRID_SPACING;
}

} // namespace

QHash<QString, QPointF> arrange(const QVector<Node> &nodes,
                                const QVector<Edge> &edges,
                                FlowDirection direction) {
    QHash<QString, QPointF> positions;
    if (nodes.isEmpty()) return positions;

    QHash<QString, size_t> indexById;
    for (size_t i = 0; i < size_t(nodes.size()); ++i)
        indexById.insert(nodes[int(i)].id, i);

    QVector<QVector<size_t>> next(nodes.size()), previous(nodes.size());
    QVector<size_t> indegree(nodes.size(), 0);
    for (const Edge &edge : edges) {
        const auto source = indexById.constFind(edge.source);
        const auto target = indexById.constFind(edge.target);
        if (source == indexById.cend() || target == indexById.cend()) continue;
        next[int(*source)].push_back(*target);
        previous[int(*target)].push_back(*source);
        ++indegree[int(*target)];
    }

    QVector<int> ranks(nodes.size(), 0);
    std::set<std::pair<QString, size_t>> ready;
    for (size_t i = 0; i < size_t(nodes.size()); ++i)
        if (indegree[int(i)] == 0) ready.emplace(nodes[int(i)].id, i);
    while (!ready.empty()) {
        const size_t source = ready.begin()->second;
        ready.erase(ready.begin());
        for (size_t target : next[int(source)]) {
            ranks[int(target)] = qMax(ranks[int(target)], ranks[int(source)] + 1);
            if (--indegree[int(target)] == 0) ready.emplace(nodes[int(target)].id, target);
        }
    }
    // Valid C graphs are DAGs. Keeping malformed leftovers together still gives
    // the caller a finite, deterministic preview instead of looping.
    for (size_t i = 0; i < size_t(nodes.size()); ++i)
        if (indegree[int(i)] != 0) ranks[int(i)] = 0;

    int lastComputeRank = 0;
    for (size_t i = 0; i < size_t(nodes.size()); ++i)
        if (!nodes[int(i)].terminal) lastComputeRank = qMax(lastComputeRank, ranks[int(i)]);
    int lastRank = lastComputeRank;
    for (size_t i = 0; i < size_t(nodes.size()); ++i) {
        if (nodes[int(i)].terminal) ranks[int(i)] = qMax(ranks[int(i)], lastComputeRank + 1);
        lastRank = qMax(lastRank, ranks[int(i)]);
    }

    QVector<QVector<size_t>> layers(lastRank + 1);
    for (size_t i = 0; i < size_t(nodes.size()); ++i) layers[ranks[int(i)]].push_back(i);
    for (QVector<size_t> &layer : layers)
        std::sort(layer.begin(), layer.end(), [&nodes](size_t a, size_t b) {
            return nodes[int(a)].id < nodes[int(b)].id;
        });

    for (int sweep = 0; sweep < 4; ++sweep) {
        for (int layer = 1; layer < layers.size(); ++layer) {
            QHash<size_t, qreal> order;
            for (int i = 0; i < layers[layer - 1].size(); ++i)
                order.insert(layers[layer - 1][i], i);
            std::stable_sort(layers[layer].begin(), layers[layer].end(),
                [&](size_t a, size_t b) {
                    auto barycenter = [&](size_t node) {
                        qreal total = 0.0;
                        int count = 0;
                        for (size_t parent : previous[int(node)]) {
                            auto at = order.constFind(parent);
                            if (at != order.cend()) { total += *at; ++count; }
                        }
                        return count ? total / count : qreal(order.size());
                    };
                    const qreal left = barycenter(a), right = barycenter(b);
                    return left == right ? nodes[int(a)].id < nodes[int(b)].id : left < right;
                });
        }
        for (int layer = layers.size() - 1; layer > 0; --layer) {
            QHash<size_t, qreal> order;
            for (int i = 0; i < layers[layer].size(); ++i)
                order.insert(layers[layer][i], i);
            std::stable_sort(layers[layer - 1].begin(), layers[layer - 1].end(),
                [&](size_t a, size_t b) {
                    auto barycenter = [&](size_t node) {
                        qreal total = 0.0;
                        int count = 0;
                        for (size_t child : next[int(node)]) {
                            auto at = order.constFind(child);
                            if (at != order.cend()) { total += *at; ++count; }
                        }
                        return count ? total / count : qreal(order.size());
                    };
                    const qreal left = barycenter(a), right = barycenter(b);
                    return left == right ? nodes[int(a)].id < nodes[int(b)].id : left < right;
                });
        }
    }

    QVector<qreal> flowSizes(layers.size(), 0.0);
    for (int layer = 0; layer < layers.size(); ++layer)
        for (size_t index : layers[layer]) {
            const QRectF bounds = nodes[int(index)].bounds;
            flowSizes[layer] = qMax(flowSizes[layer],
                direction == FlowDirection::Vertical ? bounds.height() : bounds.width());
        }
    QVector<qreal> flowOffsets(layers.size(), 0.0);
    for (int layer = 1; layer < layers.size(); ++layer) {
        int tracks = 0;
        for (size_t source = 0; source < size_t(nodes.size()); ++source)
            for (size_t target : next[int(source)])
                if (ranks[int(source)] < layer && ranks[int(target)] >= layer) ++tracks;
        const qreal rawCorridor = RankGap + qMin(tracks, 24) * 16.0;
        const qreal corridor = std::ceil(rawCorridor / NN_MODEL_GRID_SPACING) *
                               NN_MODEL_GRID_SPACING;
        flowOffsets[layer] = flowOffsets[layer - 1] +
            std::ceil(flowSizes[layer - 1] / NN_MODEL_GRID_SPACING) * NN_MODEL_GRID_SPACING + corridor;
    }
    const qreal totalFlow = flowOffsets.back() +
        std::ceil(flowSizes.back() / NN_MODEL_GRID_SPACING) * NN_MODEL_GRID_SPACING;

    QHash<size_t, qreal> crossAnchors;
    for (int layer = 0; layer < layers.size(); ++layer) {
        std::map<qreal, QVector<size_t>> groups;
        qreal rootSpan = 0.0;
        if (layer == 0) {
            for (size_t index : layers[layer]) {
                const QRectF bounds = nodes[int(index)].bounds;
                rootSpan += direction == FlowDirection::Vertical ? bounds.width() : bounds.height();
            }
            if (layers[layer].size() > 1) rootSpan += (layers[layer].size() - 1) * LaneGap;
        }
        const qreal unconnectedAnchor = layer == 0 ? Margin + rootSpan / 2.0 : Margin;
        for (size_t index : layers[layer]) {
            qreal ideal = unconnectedAnchor;
            qreal total = 0.0;
            int count = 0;
            for (size_t parent : previous[int(index)]) {
                const auto anchor = crossAnchors.constFind(parent);
                if (anchor != crossAnchors.cend()) { total += *anchor; ++count; }
            }
            if (count) ideal = total / count;
            groups[ideal].push_back(index);
        }

        qreal previousRight = -std::numeric_limits<qreal>::infinity();
        for (auto &group : groups) {
            qreal cursor = 0.0;
            QVector<qreal> baseOrigins;
            qreal groupLeft = std::numeric_limits<qreal>::infinity();
            qreal groupRight = -std::numeric_limits<qreal>::infinity();
            qreal anchorTotal = 0.0;
            for (size_t index : group.second) {
                const QRectF bounds = nodes[int(index)].bounds;
                const qreal low = direction == FlowDirection::Vertical ? bounds.left() : bounds.top();
                const qreal high = direction == FlowDirection::Vertical ? bounds.right() : bounds.bottom();
                const qreal origin = cursor - low;
                baseOrigins.push_back(origin);
                groupLeft = qMin(groupLeft, origin + low);
                groupRight = qMax(groupRight, origin + high);
                anchorTotal += origin + nodes[int(index)].crossAxisAnchor;
                cursor += high - low + LaneGap;
            }
            qreal shift = group.first - anchorTotal / group.second.size();
            if (std::isfinite(previousRight) && groupLeft + shift < previousRight + LaneGap)
                shift += previousRight + LaneGap - (groupLeft + shift);
            for (int i = 0; i < group.second.size(); ++i) {
                const size_t index = group.second[i];
                const Node &node = nodes[int(index)];
                const QRectF bounds = node.bounds;
                const qreal origin = baseOrigins[i] + shift;
                const qreal flowOrigin = direction == FlowDirection::Vertical
                    ? Margin + flowOffsets[layer] - bounds.top()
                    : Margin + totalFlow - flowOffsets[layer] - flowSizes[layer] - bounds.left();
                const qreal x = direction == FlowDirection::Vertical ? origin : flowOrigin;
                const qreal y = direction == FlowDirection::Vertical ? flowOrigin : origin;
                const QPointF snapped(snap(x), snap(y));
                positions.insert(node.id, snapped);
                crossAnchors.insert(index,
                    (direction == FlowDirection::Vertical ? snapped.x() : snapped.y()) +
                    node.crossAxisAnchor);
            }
            previousRight = groupRight + shift;
        }
    }
    return positions;
}

} // namespace DataflowLayout
