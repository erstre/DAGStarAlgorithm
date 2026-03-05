package org.dbanelas;
import java.util.*;
import java.util.stream.Collectors;

public class SignatureAssigner {

    // The fixed order of operators
    // Position 0 corresponds to the first operator
    private final List<String> operatorOrder;
    private final Map<String, Integer> operatorToIndex;
    private final Map<Configuration, Character> configToSymbol;
    private final Map<Character, Configuration> symbolToConfig;
    private static final char UNASSIGNED_SYMBOL = '_';
    private static final char START_END_SYMBOL = '+';


    // Start at 33 ('!') (printable characters)
    private static final int CHAR_OFFSET = 33;

    public SignatureAssigner(Workflow graph) {
        this.operatorOrder = new ArrayList<>(graph.getNodes().keySet());
        this.operatorToIndex = new HashMap<>();
        this.configToSymbol = new HashMap<>();
        this.symbolToConfig = new HashMap<>();

        // Sort the operators based on their name
        Collections.sort(operatorOrder);

        // Map each operator to its index
        for (int i = 0; i < operatorOrder.size(); i++) {
            operatorToIndex.put(operatorOrder.get(i), i);
        }



        Set<Configuration> totalConfigs = new HashSet<>();
        for (Operator operator : graph.getNodes().values()) {
            Set<Configuration> configIds = new HashSet<>(operator.validConfigurations());
            totalConfigs.addAll(configIds);
        }

        // Counter for generating unique symbols
        int charCounter = CHAR_OFFSET;
        for (Configuration config : totalConfigs) {
            // Check for overflow (Java char is 16-bit unsigned, max 65535)
            if (charCounter > 65535) {
                throw new IllegalStateException("Too many total configurations! Exceeded 16-bit char limit.");
            }

            // Avoid the UNASSIGNED_SYMBOL in our generated alphabet
            if ((char) charCounter == UNASSIGNED_SYMBOL) {
                charCounter++;
            }

            // Avoid the START_END_SYMBOL in our generated alphabet
            if ((char) charCounter == START_END_SYMBOL) {
                charCounter++;
            }

            configToSymbol.put(config, (char) charCounter);
            symbolToConfig.put((char) charCounter, config);
            charCounter++;
        }
    }

    public Configuration getConfigurationFromSymbol(char symbol) {
        if (symbol == UNASSIGNED_SYMBOL || symbol == START_END_SYMBOL) { // TODO: FIX
            return new Configuration("dummy", "dummy");
        }
        return symbolToConfig.get(symbol);
    }

    /**
     * Generates a unique plan signature based on the assignments provided.
     * @return A String where the i-th index represents the configuration of the i-th operator.
     */
    public String generateSignature(Map<String, Configuration> assignments) {
        // Create the char array
        char[] signatureChars = new char[operatorOrder.size()];
        Arrays.fill(signatureChars, UNASSIGNED_SYMBOL);

        // Fill in the known assignments
        for (Map.Entry<String, Configuration> entry : assignments.entrySet()) {
            String operatorId = entry.getKey();
            Configuration config = entry.getValue();

            if (operatorId.equals(Workflow.START_NODE_ID) || operatorId.equals(Workflow.END_NODE_ID)) {
                // Special symbol for start/end nodes
                Integer index = operatorToIndex.get(operatorId);
                if (index != null) {
                    signatureChars[index] = START_END_SYMBOL;
                }
                continue;
            }

            Integer index = operatorToIndex.get(operatorId);
            if (index != null) {
                // Get the symbol for this config
                Character symbol = configToSymbol.get(config.getID());
                if (symbol != null) {
                    signatureChars[index] = symbol;
                }
            }
        }

        // Return as immutable String
        return new String(signatureChars);
    }
}