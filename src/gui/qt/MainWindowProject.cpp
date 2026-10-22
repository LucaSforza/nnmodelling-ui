#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "project/project.h"
#include <QApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QPushButton>
#include <QStatusBar>
#include <QVBoxLayout>

using namespace MainWindowUtils;

bool MainWindow::openProject(const QString &directory) {
    if (!confirmReplaceProject()) return false;
    const QByteArray path = QDir::cleanPath(directory).toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_open(application_.get(), path.constData(), error, sizeof(error))) {
        const QString reason = QString::fromUtf8(error);
        QMessageBox::critical(this, tr("Open failed"),
            tr("'%1' is not a valid NNModelling project. Choose a project folder containing model.json.\n\nDetails: %2")
                .arg(QDir::toNativeSeparators(directory), reason));
        return false;
    }
    scene_->clearSelection();
    scene_->setScope(QString());
    refreshAll();
    view_->fitGraph();
    statusBar()->showMessage(tr("Opened %1").arg(directory), 4000);
    return true;
}

bool MainWindow::confirmReplaceProject() {
    const NNProject *project = nn_app_project(application_.get());
    if (!project || !nn_project_dirty(project)) return true;
    const auto choice = QMessageBox::question(
        this, tr("Unsaved changes"), tr("Save changes before replacing this project?"),
        QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
    if (choice == QMessageBox::Cancel) return false;
    return choice == QMessageBox::Discard || saveProject();
}

bool MainWindow::saveProject() {
    if (!nn_app_project(application_.get())) {
        statusBar()->showMessage(tr("No project is open"), 4000);
        return false;
    }
    char error[ErrorCapacity] = {};
    if (!nn_app_save(application_.get(), error, sizeof(error))) {
        QMessageBox::critical(this, tr("Save failed"), QString::fromUtf8(error));
        return false;
    }
    updateWindowTitle();
    statusBar()->showMessage(tr("Project saved"), 3000);
    return true;
}

void MainWindow::showProjectChooser(bool startup) {
    if (startup) hide();
    for (;;) {
        QMessageBox chooser(this);
        chooser.setWindowTitle(tr("NNModelling"));
        chooser.setText(tr("Create a project from a template or open an existing project."));
        QPushButton *create = chooser.addButton(tr("Create project"), QMessageBox::ActionRole);
        QPushButton *open = chooser.addButton(tr("Open project"), QMessageBox::ActionRole);
        chooser.addButton(QMessageBox::Close);
        chooser.exec();

        if (chooser.clickedButton() == create) {
            createProject();
            if (!startup || nn_app_project(application_.get())) break;
            continue;
        }
        if (chooser.clickedButton() == open) {
            const QString directory = QFileDialog::getExistingDirectory(this, tr("Open project"));
            if (!directory.isEmpty() && openProject(directory)) break;
            if (!startup) return;
            continue;
        }
        if (startup) QApplication::quit();
        return;
    }
    if (!startup) return;
    show();
    raise();
    activateWindow();
}

void MainWindow::createProject(const QString &initialTemplate) {
    if (!confirmReplaceProject()) return;
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Create project from template"));
    auto *form = new QFormLayout;
    auto *templateChoice = new QComboBox(&dialog);
    templateChoice->addItem(tr("MNIST MLP"), QStringLiteral("mnist-mlp"));
    templateChoice->addItem(tr("MNIST VAE"), QStringLiteral("mnist-vae"));
    templateChoice->addItem(tr("tiny LLM"), QStringLiteral("tiny-llm"));
    if (!initialTemplate.isEmpty())
        templateChoice->setCurrentIndex(templateChoice->findData(initialTemplate));
    auto *parent = new QLineEdit(QDir::homePath(), &dialog);
    parent->setObjectName(QStringLiteral("projectParent"));
    auto *browse = new QPushButton(tr("Browse…"), &dialog);
    browse->setObjectName(QStringLiteral("projectBrowse"));
    auto *pathRow = new QWidget(&dialog);
    auto *pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->addWidget(parent, 1);
    pathLayout->addWidget(browse);
    auto *id = new QLineEdit(QStringLiteral("mnist-mlp"), &dialog);
    id->setObjectName(QStringLiteral("projectId"));
    auto *name = new QLineEdit(tr("MNIST MLP"), &dialog);
    name->setObjectName(QStringLiteral("projectName"));
    auto *version = new QLineEdit(QStringLiteral("0.1.0"), &dialog);
    version->setObjectName(QStringLiteral("projectVersion"));
    auto *destination = new QLabel(&dialog);
    destination->setObjectName(QStringLiteral("projectDestination"));
    destination->setWordWrap(true);
    destination->setTextInteractionFlags(Qt::TextSelectableByMouse);
    form->addRow(tr("Template"), templateChoice);
    form->addRow(tr("Parent directory"), pathRow);
    form->addRow(tr("Model ID"), id);
    form->addRow(tr("Display name"), name);
    form->addRow(tr("Initial version"), version);
    form->addRow(tr("Destination"), destination);
    auto *layout = new QVBoxLayout(&dialog);
    layout->addLayout(form);
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    buttons->button(QDialogButtonBox::Ok)->setText(tr("Create"));
    layout->addWidget(buttons);
    const auto updateDefaults = [=](int index) {
        const QString kind = templateChoice->itemData(index).toString();
        const QString defaultId = kind == QStringLiteral("tiny-llm") ? QStringLiteral("mini-llm") : kind;
        const QString defaultName = kind == QStringLiteral("mnist-mlp") ? tr("MNIST MLP") :
            kind == QStringLiteral("mnist-vae") ? tr("MNIST VAE") : tr("tiny LLM");
        id->setText(defaultId);
        name->setText(defaultName);
    };
    const auto updateDestination = [=] {
        destination->setText(QDir::toNativeSeparators(QDir(parent->text()).filePath(id->text())));
    };
    connect(templateChoice, qOverload<int>(&QComboBox::currentIndexChanged), &dialog, updateDefaults);
    connect(parent, &QLineEdit::textChanged, &dialog, updateDestination);
    connect(id, &QLineEdit::textChanged, &dialog, updateDestination);
    connect(browse, &QPushButton::clicked, &dialog, [&dialog, parent] {
        auto *picker = new QFileDialog(&dialog, tr("Choose project parent directory"), parent->text());
        picker->setFileMode(QFileDialog::Directory);
        picker->setOption(QFileDialog::ShowDirsOnly);
        connect(picker, &QFileDialog::fileSelected, &dialog, [parent](const QString &path) {
            parent->setText(path);
        });
        picker->open();
    });
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    updateDefaults(templateChoice->currentIndex());
    updateDestination();
    if (dialog.exec() != QDialog::Accepted) return;
    const QString templateId = templateChoice->currentData().toString();
    const QString rawParentPath = parent->text().trimmed();
    const QString parentPathText = QDir::cleanPath(rawParentPath);
    const QString projectIdText = id->text().trimmed();
    const QString projectNameText = name->text().trimmed();
    const QString projectVersionText = version->text().trimmed();
    if (rawParentPath.isEmpty() || projectIdText.isEmpty() || projectNameText.isEmpty() || projectVersionText.isEmpty()) {
        QMessageBox::warning(this, tr("Create project"), tr("Parent directory, model ID, display name and version are required."));
        return;
    }
    if (QFileInfo::exists(QDir(parentPathText).filePath(projectIdText))) {
        QMessageBox::warning(this, tr("Create project"), tr("Destination already exists: %1").arg(destination->text()));
        return;
    }
    const QByteArray parentPath = parentPathText.toUtf8();
    const QByteArray projectId = projectIdText.toUtf8();
    const QByteArray projectName = projectNameText.toUtf8();
    const QByteArray projectVersion = projectVersionText.toUtf8();
    char error[ErrorCapacity] = {};
    const bool created = templateId == QStringLiteral("mnist-mlp")
        ? nn_app_create(application_.get(), parentPath.constData(), projectId.constData(),
                        projectName.constData(), projectVersion.constData(), true, error, sizeof(error))
        : templateId == QStringLiteral("mnist-vae")
            ? nn_app_create_vae(application_.get(), parentPath.constData(), projectId.constData(),
                                projectName.constData(), projectVersion.constData(), error, sizeof(error))
            : nn_app_create_llm(application_.get(), parentPath.constData(), projectId.constData(),
                                projectName.constData(), projectVersion.constData(), error, sizeof(error));
    if (!created) {
        QMessageBox::critical(this, tr("Create failed"), QString::fromUtf8(error));
        return;
    }
    scene_->clearSelection();
    scene_->setScope(QString());
    refreshAll();
    view_->fitGraph();
    statusBar()->showMessage(tr("Created %1").arg(projectNameText), 4000);
}
