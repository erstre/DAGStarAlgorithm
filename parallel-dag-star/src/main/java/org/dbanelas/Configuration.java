package org.dbanelas;

// Represents a deployment option for a specific operator (e.g., "Run on Device 1")
// [cite: 98] Each configuration has a cost.
public class Configuration {

    String deviceID;
    String platformID;

    public Configuration(String deviceID, String platformID) {
        this.deviceID = deviceID;
        this.platformID = platformID;
    }

    public String deviceID() {
        return deviceID;
    }

    public String platformID() {
        return platformID;
    }

    public static String getConfigurationID(String deviceID, String platformID) {
        return deviceID + "_" + platformID;
    }

    public String getID() {
        return deviceID + "_" + platformID;
    }

    // Implement equals and hashCode for proper comparison and usage in collections
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Configuration)) return false;
        Configuration that = (Configuration) o;
        return deviceID.equals(that.deviceID) && platformID.equals(that.platformID);
    }

    @Override
    public int hashCode() {
        int result = deviceID.hashCode();
        result = 31 * result + platformID.hashCode();
        return result;
    }

    // Add toString method for better readability
    @Override
    public String toString() {
        return "Configuration{" +
                "deviceID='" + deviceID + '\'' +
                ", platformID='" + platformID + '\'' +
                '}';

    }
}
