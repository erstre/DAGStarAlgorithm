package org.dbanelas.parsers;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.dbanelas.cost.DAGStarCostModel;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;

public class CostModelParser {

    private static final String DEFAULT_CSV_PLATFORM = "platform_0";

    /**
     * Factory method to create and populate a DAGStarCostModel from strings.
     */
    public static DAGStarCostModel load(InputStream costFileInputStream, String costFileTitle, String networkContents) {
        DAGStarCostModel model = new DAGStarCostModel();

        parseProcessingCosts(model, costFileInputStream, costFileTitle);
        parseNetworkCosts(model, networkContents);

        return model;
    }

    private static void parseProcessingCosts(DAGStarCostModel model,
                                             InputStream xlsxInputStream,
                                             String xlsxName) {
        try (Workbook workbook = new XSSFWorkbook(xlsxInputStream)) {
            // Assume data is on the first sheet
            Sheet sheet = workbook.getSheetAt(0);

            // Iterate over rows
            // Using a for-loop based on physical indexes is often safer for structured data than an iterator
            int totalRows = sheet.getPhysicalNumberOfRows();
            if (totalRows < 2) return;

            // 1. Parse Header (Row 0)
            Row headerRow = sheet.getRow(0);
            List<String> colIndexToDevice = new ArrayList<>();

            // Start from column 1 (skipping the "Operator" column header)
            // getLastCellNum() returns the index + 1 (1-based)
            for (int i = 1; i < headerRow.getLastCellNum(); i++) {
                Cell cell = headerRow.getCell(i);
                if (cell != null) {
                    colIndexToDevice.add(cell.getStringCellValue().trim());
                }
            }

            // 2. Parse Data Rows (Row 1 to End)
            for (int i = 1; i < totalRows; i++) {
                Row row = sheet.getRow(i);
                if (row == null) continue;

                // Get Operator Name (Column 0)
                Cell operatorCell = row.getCell(0);
                if (operatorCell == null) continue;

                String operator = operatorCell.getStringCellValue().trim();

                // Get Costs (from column one and onwards)
                for (int j = 1; j < row.getLastCellNum() && (j - 1) < colIndexToDevice.size(); j++) {
                    Cell costCell = row.getCell(j);
                    if (costCell != null) {
                        try {
                            double cost;
                            if (costCell.getCellType() == CellType.NUMERIC) {
                                cost = costCell.getNumericCellValue();
                            } else {
                                cost = Double.parseDouble(costCell.getStringCellValue().trim());
                            }

                            String device = colIndexToDevice.get(j - 1);

                            // Inject data into the model
                            model.addProcessingCost(operator, device, DEFAULT_CSV_PLATFORM, cost);
                        } catch (NumberFormatException | IllegalStateException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void parseNetworkCosts(DAGStarCostModel model, String txtContent) {
        String[] lines = txtContent.split("\\r?\\n");
        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("[")) continue;

            String[] parts = line.split("=");
            if (parts.length != 2) continue;

            try {
                double latency = Double.parseDouble(parts[1].trim());
                String[] devicesPair = parts[0].split(":");

                if (devicesPair.length == 2) {
                    model.addNetworkLatency(devicesPair[0].trim(), devicesPair[1].trim(), latency);
                }
            } catch (NumberFormatException e) {
                // Ignore malformed lines
            }
        }
    }
}