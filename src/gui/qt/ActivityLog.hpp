#ifndef NN_GUI_ACTIVITY_LOG_HPP
#define NN_GUI_ACTIVITY_LOG_HPP

#include "activity/activity.h"

#include <QFileDialog>
#include <QJsonObject>
#include <QMessageBox>
#include <QString>

#include <functional>

class QAction;
class QDialog;
class QObject;
class QWidget;

// Qt side of the anonymous activity log (docs/knowledge/contracts/activity-log.md).
// Borrows one process-wide NNActivityLog; every function is a no-op without it and
// logging never alters application behavior. Application thread only.
namespace ActivityLog {

void setLog(NNActivityLog *log);
bool enabled();
// Emits a src:"ui" record. (Named record because Qt defines the emit macro.)
void record(const char *event, const QJsonObject &data = {}, bool ok = true,
            const QString &error = {});
// Same, but skipped when the previous record of this event had identical data.
void recordChanged(const char *event, const QJsonObject &data);
// Replaces the $HOME prefix with "~" in free text such as messages and paths.
QString redact(const QString &text);

// Runs fn once, after the event loop turn ends, however often it is requested.
void coalesce(QObject *context, const char *key, std::function<void()> fn);
// Runs fn after ms without a further request for the same key.
void debounce(QObject *context, const char *key, int ms, std::function<void()> fn);

// Records ui.action when the action triggers; via is derived from the action.
void watchAction(QAction *action);
// Suppresses ui.action records while alive (for actions that forward to another action).
class Suppress {
public:
    Suppress();
    ~Suppress();
    Suppress(const Suppress &) = delete;
    Suppress &operator=(const Suppress &) = delete;
};

void dialogOpen(const char *dialog);
void dialogClose(const char *dialog, const char *result);
// QDialog::exec with dialog.open/dialog.close records; returns exec()'s result.
int exec(const char *dialog, QDialog &dialog_);
// Message boxes. warning/critical with only an Ok button also record error.shown.
QMessageBox::StandardButton warning(QWidget *parent, const QString &title, const QString &text,
    QMessageBox::StandardButtons buttons = QMessageBox::Ok,
    QMessageBox::StandardButton defaultButton = QMessageBox::NoButton,
    const char *dialog = "message");
QMessageBox::StandardButton critical(QWidget *parent, const QString &title, const QString &text,
    QMessageBox::StandardButtons buttons = QMessageBox::Ok,
    QMessageBox::StandardButton defaultButton = QMessageBox::NoButton,
    const char *dialog = "message");
QMessageBox::StandardButton question(const char *dialog, QWidget *parent, const QString &title,
    const QString &text, QMessageBox::StandardButtons buttons,
    QMessageBox::StandardButton defaultButton);
// File chooser wrappers; result is accepted when a path was chosen.
QString existingDirectory(const char *dialog, QWidget *parent, const QString &caption,
    const QString &directory = {}, QFileDialog::Options options = QFileDialog::ShowDirsOnly);
QString saveFileName(const char *dialog, QWidget *parent, const QString &caption,
    const QString &directory = {});

}

#endif
