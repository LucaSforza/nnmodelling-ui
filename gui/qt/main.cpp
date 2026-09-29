#include "MainWindow.hpp"

#include "application.h"

#include <QApplication>
#include <QByteArray>
#include <QDir>
#include <QDebug>
#include <QTimer>

#ifndef NN_SOURCE_DIR
#define NN_SOURCE_DIR "."
#endif

int main(int argc, char **argv) {
    QApplication application(argc, argv);
    QApplication::setApplicationName(QStringLiteral("NNModelling"));
    QApplication::setOrganizationName(QStringLiteral("NNModelling"));

    QString projectPath;
    QString capturePath;
    bool exitAfterCapture = qEnvironmentVariableIsSet("NN_UI_QA_EXIT_AFTER_CAPTURE");
    for (int i = 1; i < argc; ++i) {
        const QString arg = QString::fromLocal8Bit(argv[i]);
        if (arg == QStringLiteral("--capture") && i + 1 < argc) {
            capturePath = QString::fromLocal8Bit(argv[++i]);
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
    window.show();
    if (!projectPath.isEmpty()) {
        QTimer::singleShot(0, &window, [&window, projectPath] { window.openProject(projectPath); });
    } else if (capturePath.isEmpty()) {
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
