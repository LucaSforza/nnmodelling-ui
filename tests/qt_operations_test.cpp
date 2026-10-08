#include "MainWindow.hpp"

#include "application/application.h"

#include <QAction>
#include <QApplication>
#include <QComboBox>
#include <QDialog>
#include <QLineEdit>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QTemporaryDir>
#include <QTimer>
#include <QTreeWidget>

#include <cassert>

int main(int argc, char **argv) {
    QApplication qt(argc, argv);
    QTemporaryDir parent;
    assert(parent.isValid());
    NNApplication *application = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    assert(application);
    char error[512] = {};
    assert(nn_app_create(application, parent.path().toUtf8().constData(), "operations-ui-test",
                         "Operations UI Test", false, error, sizeof(error)));
    assert(nn_app_add_node(application, "source", "core.input", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_add_node(application, "layer", "core.relu", "0.1.0", "", 100, 0,
                           error, sizeof(error)));
    char inputHandle[128] = {};
    char outputHandle[128] = {};
    assert(nn_app_port_id(application, "source", true, 0, inputHandle, sizeof(inputHandle)));
    assert(nn_app_port_id(application, "layer", true, 0, outputHandle, sizeof(outputHandle)));
    const QByteArray operations = QByteArray("[{\"name\":\"encode\",\"input\":{\"node\":\"source\",\"handle\":\"") +
        inputHandle + "\",\"codec\":\"dataset\"},\"output\":{\"node\":\"layer\",\"handle\":\"" +
        outputHandle + "\",\"codec\":\"tensor\"}},{\"name\":\"wrongDirection\",\"input\":{\"node\":\"layer\",\"handle\":\"" +
        outputHandle + "\",\"codec\":\"tensor\"},\"output\":{\"node\":\"layer\",\"handle\":\"" +
        outputHandle + "\",\"codec\":\"tensor\"}}]";
    assert(nn_app_set_operations_json(application, operations.constData(), error, sizeof(error)));

    MainWindow window(application);
    assert(window.findChild<QAction *>(QStringLiteral("manageOperationsAction")));
    QTreeWidget *resources = window.findChild<QTreeWidget *>(QStringLiteral("resources"));
    assert(resources);
    bool foundGroup = false;
    for (int i = 0; i < resources->topLevelItemCount(); ++i)
        foundGroup = foundGroup || resources->topLevelItem(i)->text(0) == QStringLiteral("Operations");
    assert(foundGroup);
    QTreeWidgetItem *operationGroup = nullptr;
    for (int i = 0; i < resources->topLevelItemCount(); ++i)
        if (resources->topLevelItem(i)->text(0) == QStringLiteral("Operations"))
            operationGroup = resources->topLevelItem(i);
    assert(operationGroup && operationGroup->childCount() == 2);
    assert(operationGroup->child(0)->text(0).contains(QStringLiteral("Ready")));
    assert(operationGroup->child(1)->text(0).contains(QStringLiteral("Stale endpoint")));

    QTimer::singleShot(0, [&] {
        auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        assert(dialog);
        auto *list = dialog->findChild<QListWidget *>(QStringLiteral("operationsList"));
        assert(list && list->count() == 2 && list->item(0)->text().contains(QStringLiteral("encode")));
        assert(list->item(1)->text().contains(QStringLiteral("Stale endpoint")));
        list->setCurrentRow(0);
        assert(dialog->findChild<QPushButton *>(QStringLiteral("newOperation")));
        assert(dialog->findChild<QPushButton *>(QStringLiteral("removeOperation")));
        auto *edit = dialog->findChild<QPushButton *>(QStringLiteral("editOperation"));
        assert(edit);
        QTimer::singleShot(0, dialog, [] {
            auto *form = qobject_cast<QDialog *>(QApplication::activeModalWidget());
            assert(form);
            assert(form->findChild<QLineEdit *>(QStringLiteral("operationName"))->text() == QStringLiteral("encode"));
            assert(form->findChild<QComboBox *>(QStringLiteral("operationInputNode"))->count() == 2);
            assert(form->findChild<QComboBox *>(QStringLiteral("operationOutputNode"))->count() == 1);
            assert(form->findChild<QLabel *>(QStringLiteral("operationSignature")));
            form->reject();
        });
        edit->click();
        QTimer::singleShot(0, dialog, [] {
            auto *form = qobject_cast<QDialog *>(QApplication::activeModalWidget());
            assert(form);
            assert(form->findChild<QLineEdit *>(QStringLiteral("operationName")));
            assert(form->findChild<QComboBox *>(QStringLiteral("operationInputNode"))->count() == 2);
            assert(form->findChild<QComboBox *>(QStringLiteral("operationOutputNode"))->count() == 1);
            assert(form->findChild<QComboBox *>(QStringLiteral("operationInputCodec"))->count() == 2);
            assert(form->findChild<QComboBox *>(QStringLiteral("operationOutputCodec"))->count() == 2);
            form->reject();
        });
        dialog->findChild<QPushButton *>(QStringLiteral("newOperation"))->click();
        dialog->reject();
    });
    window.findChild<QAction *>(QStringLiteral("manageOperationsAction"))->trigger();
    return 0;
}
