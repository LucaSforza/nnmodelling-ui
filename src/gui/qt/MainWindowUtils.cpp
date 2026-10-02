#include "MainWindowUtils.hpp"
#include "catalog/catalog.h"

#include <QTableWidgetItem>
#include <cmath>

namespace MainWindowUtils {

QString textOr(QString value, const char *fallback) {
    return value.isEmpty() ? QString::fromUtf8(fallback) : value;
}

QString packageColor(const NNPackage *package) {
    QString color = package && package->color ? QString::fromUtf8(package->color) : QString();
    if (color.startsWith('#') && (color.size() == 4 || color.size() == 7)) return color;
    return QStringLiteral("#6b8fc4");
}

static qreal relativeLuminance(const QColor &color) {
    auto channel = [](int value) {
        const qreal c = value / 255.0;
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * channel(color.red()) + 0.7152 * channel(color.green()) +
           0.0722 * channel(color.blue());
}

QColor readablePackageColor(const NNPackage *package) {
    QColor color(packageColor(package));
    while (color.isValid() && (1.05 / (relativeLuminance(color) + 0.05)) < 4.5)
        color = color.darker(110);
    return color;
}

void addTableRow(QTableWidget *table, const QStringList &defaults) {
    const int row = table->rowCount();
    table->insertRow(row);
    for (int column = 0; column < defaults.size(); ++column) {
        auto *cell = new QTableWidgetItem(defaults[column]);
        table->setItem(row, column, cell);
    }
}

QString cellText(const QTableWidget *table, int row, int column) {
    const auto *item = table->item(row, column);
    return item ? item->text().trimmed() : QString();
}

}
