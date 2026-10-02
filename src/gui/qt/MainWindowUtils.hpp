#ifndef NN_GUI_MAIN_WINDOW_UTILS_HPP
#define NN_GUI_MAIN_WINDOW_UTILS_HPP
#include <QColor>
#include <QString>
#include <QStringList>
#include <QTableWidget>
#include <cstddef>
struct NNPackage;
namespace MainWindowUtils {
constexpr size_t ErrorCapacity = 512;
constexpr int IdRole = Qt::UserRole;
constexpr int PackageVersionRole = Qt::UserRole + 1;
constexpr int DatasetVersionRole = Qt::UserRole + 2;
QString textOr(QString value, const char *fallback);
QString packageColor(const NNPackage *package);
QColor readablePackageColor(const NNPackage *package);
void addTableRow(QTableWidget *table, const QStringList &defaults);
QString cellText(const QTableWidget *table, int row, int column);
}
#endif
