#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "inference/inference.h"
#include "model/model.h"
#include <QBrush>
#include <QCheckBox>
#include <QComboBox>
#include <QColor>
#include <QHash>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QStringList>
#include <utility>
#include <QVector>
#include <utility>

using namespace MainWindowUtils;

void MainWindow::refreshDiagnostics() {
    diagnostics_->clear();
    QHash<QString, QString> markers;
    if (!nn_app_project(application_.get())) {
        scene_->setProblemMarkers(markers);
        return;
    }
    char error[ErrorCapacity] = {};
    const NNInferenceReport *report = nn_app_analysis(application_.get(), error, sizeof(error));
    if (!report) {
        auto *unavailable = new QTreeWidgetItem(diagnostics_,
            {QStringLiteral("■  Analysis unavailable\n%1").arg(tr("Could not analyze this model."))});
        unavailable->setForeground(0, QBrush(QColor("#554d79")));
        auto *technical = new QTreeWidgetItem(unavailable, {tr("Technical details")});
        new QTreeWidgetItem(technical, {tr("Details: %1").arg(QString::fromUtf8(error))});
        scene_->setProblemMarkers(markers);
        return;
    }
    struct Problem {
        QString id, cause, category, label, scope, scopeId, message, source, code;
        size_t line = 0;
        NNInferenceStatus status = NN_INFERENCE_SUCCESS;
        bool context = false;
        bool rootProblem = false;
    };
    QVector<Problem> problems;
    const NNModel *model = nn_app_model(application_.get());
    QHash<QString, QString> labels;
    for (size_t i = 0; model && i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (node) labels.insert(QString::fromUtf8(node->id),
            textOr(QString::fromUtf8(node->label), node->id));
    }
    auto scopePath = [model, &labels](QString scope) {
        QStringList parts;
        QSet<QString> visited;
        while (!scope.isEmpty() && !visited.contains(scope)) {
            visited.insert(scope);
            parts.prepend(labels.value(scope, scope));
            const QByteArray scopeBytes = scope.toUtf8();
            const NNNode *owner = nn_model_find_node(model, scopeBytes.constData());
            scope = owner ? QString::fromUtf8(owner->scope_id ? owner->scope_id : "") : QString();
        }
        return parts.isEmpty() ? tr("Root") : parts.join(QStringLiteral(" / "));
    };
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        if (!result || result->status == NN_INFERENCE_SUCCESS || !result->node_id) continue;
        const QString id = QString::fromUtf8(result->node_id);
        const NNNode *node = nn_model_find_node(model, result->node_id);
        const QString scope = node ? QString::fromUtf8(node->scope_id ? node->scope_id : "") : QString();
        const char *categoryText = nn_inference_category(result->status);
        const QString category = QString::fromUtf8(categoryText ? categoryText : "internal");
        markers.insert(id, category);
        Problem p;
        p.id = id;
        p.cause = QString::fromUtf8(result->cause_node_id ? result->cause_node_id : "");
        p.category = category;
        p.label = labels.value(id, id);
        p.scope = scopePath(scope);
        p.scopeId = scope;
        p.message = QString::fromUtf8(result->message ? result->message : "");
        p.source = QString::fromUtf8(result->source_file ? result->source_file : "");
        p.code = QString::fromUtf8(result->code ? result->code : "");
        p.line = result->source_line;
        p.status = result->status;
        problems.push_back(p);
    }
    const NNInferenceStatus rootStatus = nn_inference_root_status(report);
    if (rootStatus != NN_INFERENCE_SUCCESS) {
        Problem root;
        const char *rootCategory = nn_inference_category(rootStatus);
        root.category = QString::fromUtf8(rootCategory ? rootCategory : "incomplete");
        root.label = tr("Root");
        root.scope = tr("Root");
        const char *rootMessage = nn_inference_root_message(report);
        root.message = rootMessage ? QString::fromUtf8(rootMessage)
                                   : tr("Root boundaries are incomplete.");
        root.status = rootStatus;
        root.rootProblem = true;
        root.code = rootStatus == NN_INFERENCE_SEMANTIC_ERROR
            ? QStringLiteral("model.semantic") : QStringLiteral("model.incomplete");
        problems.prepend(root);
    }
    if (currentScopeProblems_->isChecked()) {
        const QString currentScope = scene_->scope();
        QSet<QString> relevantIds;
        for (const Problem &p : problems) {
            if (p.scopeId != currentScope) continue;
            relevantIds.insert(p.id);
            if (!p.cause.isEmpty()) relevantIds.insert(p.cause);
        }
        QVector<Problem> relevant;
        relevant.reserve(problems.size());
        for (Problem p : problems) {
            if (p.rootProblem) {
                if (currentScope.isEmpty()) relevant.push_back(std::move(p));
                continue;
            }
            if (!relevantIds.contains(p.id)) continue;
            p.context = p.scopeId != currentScope;
            relevant.push_back(std::move(p));
        }
        problems = std::move(relevant);
    }
    auto categoryPresentation = [](const QString &category) {
        if (category == QStringLiteral("lua-compilation"))
            return qMakePair(QStringLiteral("L  Lua compilation error"), QColor("#9c3d79"));
        if (category == QStringLiteral("incomplete"))
            return qMakePair(QStringLiteral("?  Incomplete"), QColor("#a66a12"));
        if (category == QStringLiteral("internal"))
            return qMakePair(QStringLiteral("×  Analysis unavailable"), QColor("#554d79"));
        return qMakePair(QStringLiteral("!  Model error"), QColor("#b23b35"));
    };
    QHash<QString, int> byId;
    for (int i = 0; i < problems.size(); ++i) byId.insert(problems[i].id, i);
    QHash<QString, QTreeWidgetItem *> roots;
    for (const Problem &p : problems) {
        if (!p.cause.isEmpty() && byId.contains(p.cause)) continue;
        const auto presentation = categoryPresentation(p.category);
        const QString summary = p.category == QStringLiteral("internal")
            ? tr("Could not analyze this model.") : p.message;
        const QString contextSuffix = p.context ? tr(" · cause context") : QString();
        const QString displayText = QStringLiteral("%1 · %2%3\n%4: %5\n%6")
            .arg(presentation.first, p.label, contextSuffix, tr("Scope"), p.scope, summary);
        auto *root = new QTreeWidgetItem(diagnostics_,
            {displayText});
        if (!p.rootProblem) root->setData(0, IdRole, p.id);
        root->setForeground(0, QBrush(presentation.second));
        root->setToolTip(0, displayText);
        if (!p.source.isEmpty() || p.line || p.category == QStringLiteral("internal")) {
            auto *technical = new QTreeWidgetItem(root, {tr("Technical details")});
            if (!p.code.isEmpty()) new QTreeWidgetItem(technical, {tr("%1: %2").arg(tr("Code"), p.code)});
            if (!p.source.isEmpty()) new QTreeWidgetItem(technical, {tr("%1: %2").arg(tr("Source"),
                p.line ? QStringLiteral("%1:%2").arg(p.source).arg(p.line) : p.source)});
            if (p.category == QStringLiteral("internal") && !p.message.isEmpty())
                new QTreeWidgetItem(technical, {tr("%1: %2").arg(tr("Details"), p.message)});
        }
        roots.insert(p.id, root);
    }
    for (const Problem &p : problems) {
        if (p.cause.isEmpty() || !roots.contains(p.cause)) continue;
        const auto presentation = categoryPresentation(p.category);
        const QString summary = p.category == QStringLiteral("internal")
            ? tr("Analysis unavailable for this node.") : p.message;
        const QString displayText = QStringLiteral("%1 · %2 · %3\n%4: %5\n%6")
            .arg(presentation.first, p.label, tr("Blocked descendant"),
                 tr("Scope"), p.scope, summary);
        auto *child = new QTreeWidgetItem(roots.value(p.cause), {displayText});
        child->setData(0, IdRole, p.id);
        child->setForeground(0, QBrush(presentation.second));
        child->setToolTip(0, displayText);
        if (p.category == QStringLiteral("internal")) {
            auto *technical = new QTreeWidgetItem(child, {tr("Technical details")});
            if (!p.code.isEmpty()) new QTreeWidgetItem(technical, {tr("%1: %2").arg(tr("Code"), p.code)});
            if (!p.message.isEmpty()) new QTreeWidgetItem(technical,
                {tr("%1: %2").arg(tr("Details"), p.message)});
        }
    }
    if (problems.isEmpty()) {
        auto *none = new QTreeWidgetItem(diagnostics_, {currentScopeProblems_->isChecked()
            ? tr("No problems in this scope") : tr("No model problems")});
        none->setForeground(0, QBrush(QColor("#58677a")));
    }
    scene_->setProblemMarkers(markers);
}

bool MainWindow::revealNode(const QString &id) {
    const QByteArray nodeId = id.toUtf8();
    const NNModel *model = nn_app_model(application_.get());
    if (!model || !nn_model_find_node(model, nodeId.constData())) return false;
    currentScopeProblems_->setChecked(false);
    scene_->revealNode(id);
    const NNNode *node = nn_model_find_node(nn_app_model(application_.get()), nodeId.constData());
    const QString scope = QString::fromUtf8(node->scope_id ? node->scope_id : "");
    const int index = scopeSelector_->findData(scope);
    if (index >= 0) scopeSelector_->setCurrentIndex(index);
    return true;
}
