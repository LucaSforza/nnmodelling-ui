#include "MainWindow.hpp"

#include "ActivityLog.hpp"

#include "application/application.h"

#include <QApplication>
#include <QByteArray>
#include <QDir>
#include <QDebug>
#include <QFile>
#include <QGuiApplication>
#include <QJsonArray>
#include <QJsonObject>
#include <QLocale>
#include <QScreen>
#include <QStringList>
#include <QTimer>

#include <cstdio>

#ifndef NN_SOURCE_DIR
#define NN_SOURCE_DIR "."
#endif

namespace {
// PRETTY_NAME of /etc/os-release, or a null value.
QJsonValue operatingSystemName() {
    QFile file(QStringLiteral("/etc/os-release"));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return QJsonValue::Null;
    while (!file.atEnd()) {
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (!line.startsWith(QStringLiteral("PRETTY_NAME="))) continue;
        QString value = line.mid(12);
        if (value.size() >= 2 && (value.startsWith(QLatin1Char('"')) || value.startsWith(QLatin1Char('\''))))
            value = value.mid(1, value.size() - 2);
        return value;
    }
    return QJsonValue::Null;
}
}

int main(int argc, char **argv) {
    QStringList originalArguments;
    for (int i = 1; i < argc; ++i)
        originalArguments.push_back(QString::fromLocal8Bit(argv[i]));
    for (int i = 0; i < originalArguments.size(); ++i) {
        if (originalArguments.at(i) == QStringLiteral("--socket") &&
            (i + 1 >= originalArguments.size() ||
             originalArguments.at(i + 1).startsWith(QStringLiteral("--")))) {
            qWarning().noquote() << "--socket requires a socket path";
            std::fputs("--socket requires a socket path\n", stderr);
            return 2;
        }
    }
    QApplication application(argc, argv);
    QApplication::setApplicationName(QStringLiteral("NNModelling"));
    QApplication::setOrganizationName(QStringLiteral("NNModelling"));
#ifdef NN_APP_VERSION
    QApplication::setApplicationVersion(QStringLiteral(NN_APP_VERSION));
#endif

    NNActivityLog *activityLog = nullptr;
    {
        char path[4096] = {};
        char error[256] = {};
        if (nn_activity_resolve_path(path, sizeof(path), error, sizeof(error))) {
            activityLog = nn_activity_open(path, error, sizeof(error));
            if (!activityLog)
                std::fprintf(stderr, "Activity log disabled: %s\n", error);
        } else if (error[0]) {
            std::fprintf(stderr, "Activity log disabled: %s\n", error);
        }
    }

    QString projectPath;
    QString capturePath;
    QString socketPath;
    bool exitAfterCapture = qEnvironmentVariableIsSet("NN_UI_QA_EXIT_AFTER_CAPTURE");
    for (int i = 0; i < originalArguments.size(); ++i) {
        const QString arg = originalArguments.at(i);
        if (arg == QStringLiteral("--capture") && i + 1 < originalArguments.size()) {
            capturePath = originalArguments.at(++i);
        } else if (arg == QStringLiteral("--socket")) {
            socketPath = originalArguments.at(++i);
        } else if (arg == QStringLiteral("--quit-after-capture")) {
            exitAfterCapture = true;
        } else if (!arg.startsWith(QStringLiteral("--"))) {
            projectPath = arg;
        }
    }
    if (capturePath.isEmpty()) capturePath = qEnvironmentVariable("NN_UI_QA_CAPTURE");

    const QString appDir = qEnvironmentVariable("APPDIR");
    const QString resourceRoot = appDir.isEmpty() ? QString::fromUtf8(NN_SOURCE_DIR)
        : QDir(appDir).filePath(QStringLiteral("usr/share/nnmodelling"));
    const QByteArray coreRoot = QDir(resourceRoot)
        .filePath(QStringLiteral("stereotype-packages/core")).toUtf8();
    NNApplication *nativeApp = nn_app_new(coreRoot.constData());
    if (!nativeApp) {
        nn_activity_close(activityLog);
        return 1;
    }
    nn_app_set_activity_log(nativeApp, activityLog);
    ActivityLog::setLog(activityLog);
    if (activityLog) {
        const QScreen *screen = QGuiApplication::primaryScreen();
        const QSize size = screen ? screen->size() : QSize();
        const QString version = QCoreApplication::applicationVersion();
        ActivityLog::record("session.start", {
            {QStringLiteral("app"), QStringLiteral("nnmodelling-ui")},
            {QStringLiteral("version"), version.isEmpty() ? QJsonValue(QJsonValue::Null)
                                                          : QJsonValue(version)},
            {QStringLiteral("qt"), QString::fromLatin1(qVersion())},
            {QStringLiteral("os"), operatingSystemName()},
            {QStringLiteral("locale"), QLocale::system().name()},
            {QStringLiteral("platform"), QGuiApplication::platformName()},
            {QStringLiteral("screen"), QJsonArray{size.width(), size.height()}},
            {QStringLiteral("project_arg"), !projectPath.isEmpty()},
            {QStringLiteral("automation"), !socketPath.isEmpty()}});
        QObject::connect(&application, &QGuiApplication::applicationStateChanged, &application,
                         [](Qt::ApplicationState state) {
            ActivityLog::record(state == Qt::ApplicationActive ? "app.focus" : "app.blur");
        });
    }
    MainWindow window(nativeApp);
    // The window owns the application; detach the log before that is destroyed.
    const auto closeActivityLog = [nativeApp, activityLog] {
        nn_app_set_activity_log(nativeApp, nullptr);
        ActivityLog::setLog(nullptr);
        nn_activity_close(activityLog);
    };
    if (!socketPath.isEmpty() && !window.startAutomation(socketPath)) {
        closeActivityLog();
        return 1;
    }
    if (!projectPath.isEmpty()) {
        window.show();
        QTimer::singleShot(0, &window, [&window, projectPath] { window.openProject(projectPath); });
    } else if (capturePath.isEmpty() && socketPath.isEmpty()) {
        QTimer::singleShot(0, &window, [&window] { window.showProjectChooser(true); });
    } else {
        window.show();
    }
    if (!capturePath.isEmpty()) {
        QTimer::singleShot(800, &window, [&application, &window, capturePath, exitAfterCapture] {
            const bool saved = window.grab().save(capturePath);
            if (!saved) qWarning("Could not save UI capture to the requested path");
            if (exitAfterCapture) application.quit();
        });
    }
    const int exitCode = application.exec();
    ActivityLog::record("session.end", {{QStringLiteral("exit_code"), exitCode}});
    closeActivityLog();
    return exitCode;
}
