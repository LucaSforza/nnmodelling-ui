#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "MainWindowUtils.hpp"
#include "NodeItem.hpp"
#include "application/application.h"
#include "automation/automation.h"
#include "project/project.h"
#include <QApplication>
#include <QComboBox>
#include <QLayout>
#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QStatusBar>
#include <QTimer>
#include <cstdlib>
#include <cstring>
#include <cstdio>

using namespace MainWindowUtils;

bool MainWindow::startAutomation(const QString &socketPath) {
    if (automation_) return true;
    const QByteArray path = QDir::cleanPath(socketPath).toUtf8();
    char error[ErrorCapacity] = {};
    automation_ = nn_automation_start(application_.get(), path.constData(),
                                      &MainWindow::automationUiCallback, this,
                                      error, sizeof(error));
    if (!automation_) {
        statusBar()->showMessage(tr("Automation service failed: %1").arg(QString::fromUtf8(error)));
        return false;
    }
    automationTimer_ = new QTimer(this);
    automationTimer_->setInterval(15);
    connect(automationTimer_, &QTimer::timeout, this, [this] {
        if (!automation_) return;
        const NNProject *previousProject = nn_app_project(application_.get());
        const bool changed = nn_automation_poll(automation_);
        const NNProject *activeProject = nn_app_project(application_.get());
        if (previousProject != activeProject) {
            scene_->clearSelection();
            scene_->setScope(QString());
        }
        if (changed) refreshAll();
    });
    automationTimer_->start();
    statusBar()->showMessage(tr("Automation socket listening: %1").arg(socketPath), 6000);
    return true;
}

char *MainWindow::automationUiCallback(void *user, const char *operation,
                                       const char *argsJson, char *error, size_t cap) {
    auto *self = static_cast<MainWindow *>(user);
    auto fail = [error, cap](const char *message) -> char * {
        if (error && cap) std::snprintf(error, cap, "%s", message);
        return nullptr;
    };
    if (!self || !operation) return fail("Invalid UI callback");
    QJsonParseError parseError{};
    const QJsonDocument argsDoc = QJsonDocument::fromJson(QByteArray(argsJson ? argsJson : "{}"), &parseError);
    if (parseError.error != QJsonParseError::NoError || !argsDoc.isObject())
        return fail("UI arguments must be a JSON object");
    QJsonObject result;
    const QString op = QString::fromUtf8(operation);
    if (op == QStringLiteral("ui.inspect")) {
        QJsonArray widgets;
        struct WidgetInfo { QString id; QString role; QString label; };
        const QList<WidgetInfo> entries = {
            {QStringLiteral("main-window"), QStringLiteral("main-window"), self->windowTitle()},
            {QStringLiteral("palette"), QStringLiteral("package-palette"), self->tr("Packages")},
            {QStringLiteral("graph"), QStringLiteral("graph-canvas"), self->tr("Graph canvas")},
            {QStringLiteral("inspector"), QStringLiteral("node-inspector"), self->tr("Inspector")},
            {QStringLiteral("resources"), QStringLiteral("project-resources"), self->tr("Project resources")},
            {QStringLiteral("diagnostics"), QStringLiteral("diagnostics"), self->tr("Diagnostics")},
            {QStringLiteral("scopeSelector"), QStringLiteral("scope-selector"), self->tr("Scope")},
            {QStringLiteral("createStereotypeButton"), QStringLiteral("resource-action"), self->tr("New stereotype")},
            {QStringLiteral("createDatasetButton"), QStringLiteral("resource-action"), self->tr("New dataset")}};
        for (const auto &entry : entries) {
            QWidget *widget = entry.id == QStringLiteral("main-window")
                ? static_cast<QWidget *>(self) : self->findChild<QWidget *>(entry.id);
            if (!widget) continue;
            QJsonObject item;
            item.insert(QStringLiteral("id"), entry.id);
            item.insert(QStringLiteral("role"), entry.role);
            item.insert(QStringLiteral("label"), widget->accessibleName().isEmpty()
                ? entry.label : widget->accessibleName());
            widgets.append(item);
        }
        result.insert(QStringLiteral("widgets"), widgets);
        result.insert(QStringLiteral("currentScope"), self->scene_ ? self->scene_->scope() : QString());
    } else if (op == QStringLiteral("ui.scope")) {
        const QJsonValue requestedScope = argsDoc.object().value(QStringLiteral("id"));
        if (!requestedScope.isString()) return fail("Scope ID must be a string");
        const QString id = requestedScope.toString();
        if (id.isEmpty()) self->scene_->setScope(QString());
        else {
            const int index = self->scopeSelector_->findData(id);
            if (index < 0) return fail("Unknown scope ID");
            self->scene_->setScope(id);
        }
        self->refreshAll();
        result.insert(QStringLiteral("scope"), self->scene_->scope());
    } else if (op == QStringLiteral("ui.arrange")) {
        self->arrangeCurrentScope(FlowDirection::Vertical);
        result.insert(QStringLiteral("arranged"), true);
    } else if (op == QStringLiteral("ui.screenshot")) {
        const QString path = argsDoc.object().value(QStringLiteral("path")).toString();
        if (path.isEmpty()) return fail("Screenshot path is required");
        if (argsDoc.object().value(QStringLiteral("arrange")).toBool())
            self->arrangeCurrentScope(FlowDirection::Vertical, true);
        self->refreshAll();
        if (self->centralWidget() && self->centralWidget()->layout())
            self->centralWidget()->layout()->activate();
        self->layout()->activate();
        QApplication::processEvents();
        if (!self->grab().save(path)) return fail("Could not save screenshot");
        result.insert(QStringLiteral("path"), path);
        result.insert(QStringLiteral("saved"), true);
    } else if (op == QStringLiteral("ui.reveal")) {
        const QJsonValue requestedId = argsDoc.object().value(QStringLiteral("id"));
        if (!requestedId.isString()) return fail("Node ID must be a string");
        const QString id = requestedId.toString();
        if (!self->revealNode(id)) return fail("Unknown node ID");
        result.insert(QStringLiteral("id"), id);
    } else {
        return fail("Unsupported UI operation");
    }
    const QByteArray encoded = QJsonDocument(result).toJson(QJsonDocument::Compact);
    char *owned = static_cast<char *>(std::malloc(size_t(encoded.size()) + 1));
    if (!owned) return fail("Out of memory creating UI result");
    std::memcpy(owned, encoded.constData(), size_t(encoded.size()));
    owned[encoded.size()] = '\0';
    return owned;
}
