#include "MainWindow.hpp"
#include "application/application.h"
#include "project/project.h"

#include <QAction>
#include <QApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLineEdit>
#include <QPushButton>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QComboBox>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>

class MetavariablesWindowTest final : public QObject {
    Q_OBJECT
private slots:
    void declarationChangesAreStagedUntilSave()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
        QVERIFY(app);
        char error[512] = {};
        QVERIFY2(nn_app_create(app, temporary.path().toUtf8().constData(), "variables",
                              "Variables", "0.1.0", false, error, sizeof(error)), error);
        QVERIFY2(nn_app_save(app, error, sizeof(error)), error);
        MainWindow window(app);
        window.show();

        auto openAndAdd = [&window](bool save, const QString &entryName = QStringLiteral("num-heads"),
                                   const QString &entryType = QStringLiteral("integer"),
                                   const QString &entryValue = QStringLiteral("16")) {
            bool formCompleted = false;
            QTimer::singleShot(0, &window, [&, entryName, entryType, entryValue] {
                auto *manager = qobject_cast<QDialog *>(QApplication::activeModalWidget());
                QVERIFY(manager);
                auto *create = manager->findChild<QPushButton *>(QStringLiteral("newMetavariable"));
                QVERIFY(create);
                QTimer::singleShot(0, manager, [&, manager, entryName, entryType, entryValue] {
                    auto *form = qobject_cast<QDialog *>(QApplication::activeModalWidget());
                    QVERIFY(form);
                    auto *name = form->findChild<QLineEdit *>(QStringLiteral("metavariableName"));
                    auto *type = form->findChild<QComboBox *>(QStringLiteral("metavariableType"));
                    auto *value = form->findChild<QLineEdit *>(QStringLiteral("metavariableValue"));
                    QVERIFY(name && type && value);
                    name->setText(entryName);
                    type->setCurrentIndex(type->findData(entryType));
                    value->setText(entryValue);
                    form->accept();
                    formCompleted = true;
                });
                QTest::mouseClick(create, Qt::LeftButton);
                auto *buttons = manager->findChild<QDialogButtonBox *>(QStringLiteral("metavariablesButtons"));
                QVERIFY(buttons);
                QTest::mouseClick(buttons->button(save ? QDialogButtonBox::Save : QDialogButtonBox::Cancel),
                                  Qt::LeftButton);
            });
            for (QAction *action : window.findChildren<QAction *>())
                if (action->objectName() == QStringLiteral("manageMetavariablesAction")) action->trigger();
            QTRY_VERIFY(formCompleted);
        };

        openAndAdd(false);
        char *before = nn_app_metavariables_json(app, error, sizeof(error));
        QVERIFY2(before, error);
        QCOMPARE(QJsonDocument::fromJson(before).array().size(), 0);
        nn_app_free_text(before);

        openAndAdd(true);
        char *after = nn_app_metavariables_json(app, error, sizeof(error));
        QVERIFY2(after, error);
        const QJsonArray declarations = QJsonDocument::fromJson(after).array();
        nn_app_free_text(after);
        QCOMPARE(declarations.size(), 1);
        QCOMPARE(declarations.at(0).toObject().value(QStringLiteral("name")).toString(), QStringLiteral("num-heads"));
        QCOMPARE(declarations.at(0).toObject().value(QStringLiteral("type")).toString(), QStringLiteral("integer"));
        QCOMPARE(declarations.at(0).toObject().value(QStringLiteral("value")).toInt(), 16);

        // New must append even when a row is selected; a duplicate Save must leave the model untouched.
        bool duplicateRejected = false;
        QTimer::singleShot(0, &window, [&] {
            auto *manager = qobject_cast<QDialog *>(QApplication::activeModalWidget());
            QVERIFY(manager);
            auto *create = manager->findChild<QPushButton *>(QStringLiteral("newMetavariable"));
            QVERIFY(create);
            QTimer::singleShot(0, manager, [manager] {
                auto *form = qobject_cast<QDialog *>(QApplication::activeModalWidget());
                QVERIFY(form);
                form->findChild<QLineEdit *>(QStringLiteral("metavariableName"))->setText(QStringLiteral("num-heads"));
                auto *type = form->findChild<QComboBox *>(QStringLiteral("metavariableType"));
                type->setCurrentIndex(type->findData(QStringLiteral("integer")));
                form->findChild<QLineEdit *>(QStringLiteral("metavariableValue"))->setText(QStringLiteral("32"));
                form->accept();
            });
            QTest::mouseClick(create, Qt::LeftButton);
            QTimer::singleShot(0, manager, [&] {
                auto *warning = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
                QVERIFY(warning);
                duplicateRejected = warning->text().contains(QStringLiteral("duplicate"), Qt::CaseInsensitive) ||
                                    warning->text().contains(QStringLiteral("already"), Qt::CaseInsensitive);
                warning->accept();
            });
            QTest::mouseClick(manager->findChild<QDialogButtonBox *>(QStringLiteral("metavariablesButtons"))
                                  ->button(QDialogButtonBox::Save), Qt::LeftButton);
            QVERIFY(manager->isVisible());
            QTest::mouseClick(manager->findChild<QDialogButtonBox *>(QStringLiteral("metavariablesButtons"))
                                  ->button(QDialogButtonBox::Cancel), Qt::LeftButton);
        });
        for (QAction *action : window.findChildren<QAction *>())
            if (action->objectName() == QStringLiteral("manageMetavariablesAction")) action->trigger();
        QVERIFY(duplicateRejected);
        char *afterDuplicate = nn_app_metavariables_json(app, error, sizeof(error));
        QVERIFY2(afterDuplicate, error);
        QCOMPARE(QJsonDocument::fromJson(afterDuplicate).array().size(), 1);
        nn_app_free_text(afterDuplicate);

        openAndAdd(true, QStringLiteral("learning-rate"), QStringLiteral("number"), QStringLiteral("0.01"));
        after = nn_app_metavariables_json(app, error, sizeof(error));
        QVERIFY2(after, error);
        const QJsonArray twoDeclarations = QJsonDocument::fromJson(after).array();
        nn_app_free_text(after);
        QCOMPARE(twoDeclarations.size(), 2);
        QCOMPARE(twoDeclarations.at(1).toObject().value(QStringLiteral("name")).toString(), QStringLiteral("learning-rate"));
        QCOMPARE(twoDeclarations.at(1).toObject().value(QStringLiteral("value")).toDouble(), 0.01);
        QVERIFY(nn_project_dirty(nn_app_project(app)));
        QVERIFY2(nn_app_save(app, error, sizeof(error)), error);
        window.close();
    }
};

QTEST_MAIN(MetavariablesWindowTest)
#include "qt_metavariables_test.moc"
