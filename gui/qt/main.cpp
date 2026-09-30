#include "MainWindow.hpp"

#include "application.h"

#include <QApplication>
#include <QByteArray>
#include <QDir>
#include <QDebug>
#include <QStringList>
#include <QTimer>

#include <cstdio>

#ifndef NN_SOURCE_DIR
#define NN_SOURCE_DIR "."
#endif

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

    const QByteArray coreRoot = QDir(QString::fromUtf8(NN_SOURCE_DIR))
        .filePath(QStringLiteral("stereotype-packages/core")).toUtf8();
    NNApplication *nativeApp = nn_app_new(coreRoot.constData());
    if (!nativeApp) return 1;
    MainWindow window(nativeApp);
    if (!socketPath.isEmpty() && !window.startAutomation(socketPath)) return 1;
    window.show();
    if (!projectPath.isEmpty()) {
        QTimer::singleShot(0, &window, [&window, projectPath] { window.openProject(projectPath); });
    } else if (capturePath.isEmpty() && socketPath.isEmpty()) {
        QTimer::singleShot(0, &window, [&window] { window.showProjectChooser(); });
    }
    if (!capturePath.isEmpty()) {
        QTimer::singleShot(800, &window, [&application, &window, capturePath, exitAfterCapture] {
            const bool saved = window.grab().save(capturePath);
            if (!saved) qWarning("Could not save UI capture to the requested path");
            if (exitAfterCapture) application.quit();
        });
    }
    return application.exec();
}
