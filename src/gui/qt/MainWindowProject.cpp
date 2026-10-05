#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "project/project.h"
#include <QDir>
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QPushButton>
#include <QStatusBar>
#include <QLineEdit>

using namespace MainWindowUtils;

bool MainWindow::openProject(const QString &directory) {
    if (!confirmReplaceProject()) return false;
    const QByteArray path = QDir::cleanPath(directory).toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_open(application_.get(), path.constData(), error, sizeof(error))) {
        QMessageBox::critical(this, tr("Open failed"), QString::fromUtf8(error));
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

void MainWindow::showProjectChooser() {
    QMessageBox chooser(this);
    chooser.setWindowTitle(tr("NNModelling"));
    chooser.setText(tr("Choose a project to edit."));
    QPushButton *create = chooser.addButton(tr("New project"), QMessageBox::ActionRole);
    QPushButton *mnist = chooser.addButton(tr("New MNIST MLP"), QMessageBox::ActionRole);
    QPushButton *llm = chooser.addButton(tr("New mini LLM"), QMessageBox::ActionRole);
    QPushButton *open = chooser.addButton(tr("Open project"), QMessageBox::ActionRole);
    chooser.addButton(QMessageBox::Close);
    chooser.exec();
    if (chooser.clickedButton() == create) createProject(false);
    else if (chooser.clickedButton() == mnist) createProject(true);
    else if (chooser.clickedButton() == llm) createLlmProject();
    else if (chooser.clickedButton() == open) {
        const QString directory = QFileDialog::getExistingDirectory(this, tr("Open project"));
        if (!directory.isEmpty()) openProject(directory);
    }
}

void MainWindow::createProject(bool mnist) {
    if (!confirmReplaceProject()) return;
    const QString parent = QFileDialog::getExistingDirectory(
        this, tr("Choose project parent directory"));
    if (parent.isEmpty()) return;
    bool accepted = false;
    const QString id = QInputDialog::getText(this, tr("Project identity"),
        tr("Model ID (used as the new directory name):"), QLineEdit::Normal,
        mnist ? QStringLiteral("mnist-mlp") : QString(), &accepted).trimmed();
    if (!accepted || id.isEmpty()) return;
    const QString name = QInputDialog::getText(this, tr("Project name"), tr("Display name:"),
        QLineEdit::Normal, mnist ? QStringLiteral("MNIST MLP") : id, &accepted).trimmed();
    if (!accepted || name.isEmpty()) return;
    const QByteArray parentPath = QDir::cleanPath(parent).toUtf8();
    const QByteArray projectId = id.toUtf8();
    const QByteArray projectName = name.toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_create(application_.get(), parentPath.constData(), projectId.constData(),
                       projectName.constData(), mnist, error, sizeof(error))) {
        QMessageBox::critical(this, tr("Create failed"), QString::fromUtf8(error));
        return;
    }
    scene_->clearSelection();
    scene_->setScope(QString());
    refreshAll();
    view_->fitGraph();
    statusBar()->showMessage(tr("Created %1").arg(name), 4000);
}
void MainWindow::createLlmProject() {
    if (!confirmReplaceProject()) return;
    const QString parent = QFileDialog::getExistingDirectory(this, tr("Choose project parent directory"));
    if (parent.isEmpty()) return;
    bool accepted = false;
    const QString id = QInputDialog::getText(this, tr("Project identity"), tr("Model ID:"),
        QLineEdit::Normal, QStringLiteral("mini-llm"), &accepted).trimmed();
    if (!accepted || id.isEmpty()) return;
    const QString name = QInputDialog::getText(this, tr("Project name"), tr("Display name:"),
        QLineEdit::Normal, QStringLiteral("mini LLM"), &accepted).trimmed();
    if (!accepted || name.isEmpty()) return;
    const QByteArray parentBytes = QDir::cleanPath(parent).toUtf8();
    const QByteArray idBytes = id.toUtf8();
    const QByteArray nameBytes = name.toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_create_llm(application_.get(), parentBytes.constData(), idBytes.constData(),
                           nameBytes.constData(), error, sizeof(error))) {
        QMessageBox::critical(this, tr("Create mini LLM failed"), QString::fromUtf8(error));
        return;
    }
    scene_->clearSelection();
    scene_->setScope(QString());
    refreshAll();
    view_->fitGraph();
    statusBar()->showMessage(tr("Created %1").arg(name), 4000);
}
